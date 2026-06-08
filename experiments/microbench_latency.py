from __future__ import annotations

import asyncio
import json
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

import modal


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
VLLM_VERSION = "0.22.0"
KV_BLOCK_SIZE = 16
SERVER_PORT = 8000
STARTUP_TIMEOUT_S = 900
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
DECODE_TOKENS = 48
PROMPT = "Write a concise numbered list of simple facts about queueing systems."

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        "httpx==0.28.1",
        "huggingface_hub==0.36.0",
        "hf_transfer==0.1.9",
        "numpy==2.3.5",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("ee384s-vllm-latency-microbench", image=image)


def _reader_thread(proc: subprocess.Popen, log_lines: list[str], log_queue) -> None:
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        log_lines.append(line)
        log_queue.put(line)
        print(line, flush=True)


def _wait_for_server(base_url: str, proc: subprocess.Popen, log_lines: list[str]) -> None:
    import httpx

    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    with httpx.Client(timeout=5.0) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                recent_logs = "\n".join(log_lines[-80:])
                raise RuntimeError(
                    f"vLLM exited early with code {proc.returncode}\n"
                    f"last vLLM logs:\n{recent_logs}"
                )
            try:
                if client.get(f"{base_url}/v1/models").status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(2.0)
    recent_logs = "\n".join(log_lines[-120:])
    raise TimeoutError(
        f"vLLM did not become ready after {STARTUP_TIMEOUT_S}s\n"
        f"last vLLM logs:\n{recent_logs}"
    )


def _parse_kv_cache_info(log_text: str) -> dict[str, int | None]:
    block_size = KV_BLOCK_SIZE
    gpu_blocks = None
    gpu_tokens = None
    block_match = re.search(r"(?:block_size|block size)\D+(\d+)", log_text, re.I)
    if block_match:
        block_size = int(block_match.group(1))
    block_match = re.search(r"#\s*GPU blocks:\s*([\d,]+)", log_text, re.I)
    if block_match:
        gpu_blocks = int(block_match.group(1).replace(",", ""))
    token_match = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", log_text, re.I)
    if token_match:
        gpu_tokens = int(token_match.group(1).replace(",", ""))
        if gpu_blocks is None and block_size:
            gpu_blocks = gpu_tokens // block_size
    return {
        "gpu_kv_cache_blocks": gpu_blocks,
        "gpu_kv_cache_tokens": gpu_tokens,
        "kv_block_size": block_size,
    }


async def _one_stream(client, base_url: str, request_id: int) -> dict:
    payload = {
        "model": MODEL_ID,
        "prompt": f"{PROMPT}\nRequest id: {request_id}\n",
        "max_tokens": DECODE_TOKENS,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
    }
    start = time.perf_counter()
    token_times: list[float] = []
    async with client.stream(
        "POST", f"{base_url}/v1/completions", json=payload
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            text = chunk["choices"][0].get("text", "")
            if text:
                token_times.append(time.perf_counter())
    finish = time.perf_counter()
    step_deltas = [
        token_times[i] - token_times[i - 1] for i in range(1, len(token_times))
    ]
    return {
        "request_id": request_id,
        "ttft_s": token_times[0] - start if token_times else None,
        "latency_s": finish - start,
        "observed_chunks": len(token_times),
        "step_deltas_s": step_deltas,
    }


async def _run_batch(base_url: str, batch_size: int) -> dict:
    import httpx

    async with httpx.AsyncClient(timeout=None) as client:
        start = time.perf_counter()
        results = await asyncio.gather(
            *[_one_stream(client, base_url, idx) for idx in range(batch_size)]
        )
        wall = time.perf_counter() - start
    all_steps = [
        step for result in results for step in result["step_deltas_s"] if step > 0
    ]
    mean_step = sum(all_steps) / len(all_steps) if all_steps else wall / DECODE_TOKENS
    return {
        "batch_size": batch_size,
        "decode_tokens": DECODE_TOKENS,
        "wall_s": wall,
        "mean_step_latency_s": mean_step,
        "observed_step_samples": len(all_steps),
        "requests": results,
    }


def _fit_sublinear_curve(measurements: list[dict]) -> dict:
    import numpy as np

    batches = np.array([m["batch_size"] for m in measurements], dtype=float)
    y = np.array([m["mean_step_latency_s"] for m in measurements], dtype=float)
    best = None
    for alpha in np.linspace(0.05, 1.25, 241):
        x = (batches - 1.0) ** alpha
        design = np.column_stack([np.ones_like(x), x])
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        base, scale = coeffs
        pred = design @ coeffs
        sse = float(np.sum((pred - y) ** 2))
        if best is None or sse < best["sse"]:
            best = {
                "base_s": max(float(base), 0.0),
                "scale_s": max(float(scale), 0.0),
                "alpha": float(alpha),
                "sse": sse,
            }
    assert best is not None
    best["formula"] = "decode_step_s = base_s + scale_s * (batch_size - 1) ** alpha"
    return best


@app.function(gpu="A10G", timeout=1800, scaledown_window=20)
def run_microbench_remote(batch_sizes: list[int] = DEFAULT_BATCH_SIZES) -> dict:
    wall_start = time.perf_counter()
    base_url = f"http://127.0.0.1:{SERVER_PORT}"
    command = [
        "vllm",
        "serve",
        MODEL_ID,
        "--host",
        "0.0.0.0",
        "--port",
        str(SERVER_PORT),
        "--served-model-name",
        MODEL_ID,
        "--block-size",
        str(KV_BLOCK_SIZE),
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        str(max(batch_sizes)),
        "--gpu-memory-utilization",
        "0.85",
    ]
    print("starting:", " ".join(command))
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    log_lines: list[str] = []
    log_queue: queue.Queue[str] = queue.Queue()
    thread = threading.Thread(
        target=_reader_thread, args=(proc, log_lines, log_queue), daemon=True
    )
    thread.start()

    try:
        _wait_for_server(base_url, proc, log_lines)
        print("server ready")
        # One warmup request avoids including first-request overhead in the
        # fixed active-batch measurements.
        asyncio.run(_run_batch(base_url, 1))
        measurements = []
        for batch_size in batch_sizes:
            print(f"measuring active batch size {batch_size}")
            measurements.append(asyncio.run(_run_batch(base_url, batch_size)))
        fit = _fit_sublinear_curve(measurements)
        payload = {
            "model": MODEL_ID,
            "vllm_version": VLLM_VERSION,
            "gpu": "A10G",
            "block_size": KV_BLOCK_SIZE,
            "decode_tokens": DECODE_TOKENS,
            "prompt": PROMPT,
            "kv_cache": _parse_kv_cache_info("\n".join(log_lines)),
            "measurements": measurements,
            "fit": fit,
            "total_wall_s": time.perf_counter() - wall_start,
        }
        print("fit:", fit)
        return payload
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)
        print(f"vLLM process return code: {proc.returncode}")


@app.local_entrypoint()
def main(
    output: str = "results/latency_curve.json",
    batch_sizes: str = "1,2,4,8,16,32",
) -> None:
    parsed_batch_sizes = [int(item) for item in batch_sizes.split(",") if item]
    payload = run_microbench_remote.remote(parsed_batch_sizes)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output_path}")
