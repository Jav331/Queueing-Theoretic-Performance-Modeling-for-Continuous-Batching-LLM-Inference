from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
import time
from importlib.metadata import version

import modal


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
VLLM_VERSION = "0.22.0"
KV_BLOCK_SIZE = 16
SERVER_PORT = 8000
STARTUP_TIMEOUT_S = 420

# vLLM version was read with:
#   python -m pip index versions vllm
# on 2026-06-03, then pinned explicitly here for reproducible Modal builds.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        "httpx==0.28.1",
        "huggingface_hub==0.36.0",
        "hf_transfer==0.1.9",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("ee384s-vllm-smoke", image=image)


def _reader_thread(proc: subprocess.Popen, log_lines: list[str], log_queue) -> None:
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        log_lines.append(line)
        log_queue.put(line)


def _parse_kv_cache_info(log_text: str) -> dict[str, int | None]:
    block_size = KV_BLOCK_SIZE
    gpu_blocks = None
    gpu_tokens = None

    block_match = re.search(r"(?:block_size|block size)\D+(\d+)", log_text, re.I)
    if block_match:
        block_size = int(block_match.group(1))

    block_patterns = [
        r"#\s*GPU blocks:\s*([\d,]+)",
        r"num_gpu_blocks(?:_override)?\D+([\d,]+)",
        r"gpu_blocks\D+([\d,]+)",
    ]
    for pattern in block_patterns:
        match = re.search(pattern, log_text, re.I)
        if match:
            gpu_blocks = int(match.group(1).replace(",", ""))
            break

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


def _wait_for_models(base_url: str, proc: subprocess.Popen, log_queue) -> dict:
    import httpx

    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    last_error = None
    with httpx.Client(timeout=5.0) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                while not log_queue.empty():
                    print(log_queue.get())
                raise RuntimeError(f"vLLM exited before readiness with code {proc.returncode}")
            try:
                response = client.get(f"{base_url}/v1/models")
                if response.status_code == 200:
                    return response.json()
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except Exception as exc:
                last_error = repr(exc)
            time.sleep(2.0)
    raise TimeoutError(f"vLLM did not become ready: {last_error}")


def _stream_one_completion(base_url: str) -> float:
    import httpx

    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": "Reply with one short sentence confirming this smoke test works.",
            }
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": True,
    }
    start = time.perf_counter()
    first_token_time = None
    text_parts: list[str] = []
    with httpx.Client(timeout=None) as client:
        with client.stream(
            "POST", f"{base_url}/v1/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    text_parts.append(content)

    if first_token_time is None:
        raise RuntimeError("stream completed without a content token")
    ttft = first_token_time - start
    print(f"streamed text: {''.join(text_parts).strip()}")
    print(f"measured TTFT: {ttft:.3f}s")
    return ttft


@app.function(gpu="A10G", timeout=900, scaledown_window=20)
def smoke_vllm_server() -> None:
    wall_start = time.perf_counter()
    base_url = f"http://127.0.0.1:{SERVER_PORT}"

    print(f"pinned vLLM version: {VLLM_VERSION}")
    print(f"installed vLLM version in image: {version('vllm')}")
    print(f"model: {MODEL_ID}")

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
        "8",
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
        models = _wait_for_models(base_url, proc, log_queue)
        print("/v1/models:", models)

        log_text = "\n".join(log_lines)
        kv_info = _parse_kv_cache_info(log_text)
        print("KV cache info:", kv_info)

        _stream_one_completion(base_url)
    finally:
        print("tearing down vLLM server")
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)
        print(f"vLLM process return code: {proc.returncode}")
        print(f"total wall time: {time.perf_counter() - wall_start:.1f}s")


@app.local_entrypoint()
def main() -> None:
    smoke_vllm_server.remote()
