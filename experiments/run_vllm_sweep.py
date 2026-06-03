from __future__ import annotations

import argparse
import csv
import json
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import modal
import pandas as pd


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
VLLM_VERSION = "0.22.0"
SERVER_PORT = 8000
KV_BLOCK_SIZE = 16
STARTUP_TIMEOUT_S = 2400
REMOTE_WORKDIR = "/tmp/ee384s_vllm_sweep"
REMOTE_RESULTS_DIR = "/tmp/ee384s_vllm_sweep/results"
REMOTE_DATASET_PATH = "/tmp/ee384s_vllm_sweep/sharegpt_trace.jsonl"

DEFAULT_ARRIVAL_RATES = "0.5,1,2,3,4,5,6,8"
DEFAULT_BATCH_SIZES = "64,128"
DEFAULT_GPU_MEMORY_UTILS = "0.30,0.50,0.70,0.90"
# SimPy goodput is completed requests / experiment duration, with no latency SLO.
# vLLM bench still needs concrete thresholds for --goodput, so use loose SLOs
# and write SimPy-parity request_throughput to the CSV goodput column.
DEFAULT_GOODPUT_SLO = "ttft:100000000,tpot:100000000"
DEFAULT_MAX_MODEL_LEN = 2048
DEFAULT_NUM_PROMPTS = 300
DEFAULT_NUM_WARMUPS = 8
SMOKE_NUM_PROMPTS = 16
SMOKE_NUM_WARMUPS = 1

SUMMARY_COLUMNS = [
    "arrival_rate",
    "max_batch_size",
    "kv_budget",
    "n_seeds",
    "mean_ttft_mean",
    "mean_ttft_std",
    "p50_ttft_mean",
    "p50_ttft_std",
    "p95_ttft_mean",
    "p95_ttft_std",
    "p99_ttft_mean",
    "p99_ttft_std",
    "goodput_mean",
    "goodput_std",
    "blocking_probability_mean",
    "blocking_probability_std",
]

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        "httpx==0.28.1",
        "huggingface_hub==0.36.0",
        "hf_transfer==0.1.9",
        "numpy==2.3.5",
        "pandas==3.0.3",
        "pyarrow==24.0.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

vllm_cache = modal.Volume.from_name("ee384s-vllm-cache", create_if_missing=True)
hf_cache = modal.Volume.from_name("ee384s-huggingface-cache", create_if_missing=True)
app = modal.App("ee384s-vllm-sharegpt-sweep", image=image)


@dataclass(frozen=True)
class ServerConfig:
    max_num_seqs: int
    gpu_memory_utilization: float


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def parse_goodput_slo(text: str) -> list[str]:
    return [item.strip() for item in text.replace(",", " ").split() if item.strip()]


def build_server_configs(
    batch_sizes: list[int],
    gpu_memory_utils: list[float],
    smoke: bool,
) -> list[ServerConfig]:
    if smoke:
        smoke_batch = 8 if 8 in batch_sizes else batch_sizes[0]
        smoke_gpu_util = 0.85 if 0.85 in gpu_memory_utils else max(gpu_memory_utils)
        return [
            ServerConfig(
                max_num_seqs=smoke_batch,
                gpu_memory_utilization=smoke_gpu_util,
            )
        ]
    return [
        ServerConfig(max_num_seqs=batch, gpu_memory_utilization=util)
        for batch in batch_sizes
        for util in gpu_memory_utils
    ]


def load_trace_records(path: Path, num_prompts: int) -> list[dict[str, Any]]:
    if not path.exists():
        # TODO: Add a synthetic fallback only if we need harness-only debugging
        # without the canonical ShareGPT trace. Real sweeps should replay the
        # trace so vLLM decode demand matches the simulator input pool.
        raise FileNotFoundError(
            f"{path} is missing. Run experiments.preprocess_sharegpt first."
        )
    df = pd.read_parquet(path)
    required = {"request_id", "prompt_text", "prompt_len", "gen_len"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    if len(df) < num_prompts:
        raise ValueError(f"{path} has {len(df)} rows; requested {num_prompts} prompts.")
    rows = df.head(num_prompts).to_dict(orient="records")
    return [
        {
            "request_id": int(row["request_id"]),
            "prompt_text": str(row["prompt_text"]),
            "prompt_len": int(row["prompt_len"]),
            "gen_len": int(row["gen_len"]),
        }
        for row in rows
    ]


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_plan(
    *,
    arrival_rates: str,
    batch_sizes: str,
    gpu_memory_utils: str,
    smoke: bool,
) -> tuple[list[float], list[ServerConfig]]:
    arrival_rate_values = parse_float_list(arrival_rates)
    if smoke:
        arrival_rate_values = arrival_rate_values[:1]
    server_configs = build_server_configs(
        batch_sizes=parse_int_list(batch_sizes),
        gpu_memory_utils=parse_float_list(gpu_memory_utils),
        smoke=smoke,
    )
    return arrival_rate_values, server_configs


def print_dry_run(
    *,
    trace: Path,
    output: Path,
    metadata_output: Path,
    arrival_rates: list[float],
    server_configs: list[ServerConfig],
    max_model_len: int,
    num_prompts: int,
    num_warmups: int,
    goodput_slo: list[str],
) -> None:
    print("DRY RUN: no Modal function will be invoked.")
    print(f"trace: {trace}")
    if not trace.exists():
        print("TODO: data/sharegpt_trace.parquet is missing; run preprocess_sharegpt before a real sweep.")
    print(f"output: {output}")
    print(f"metadata_output: {metadata_output}")
    print(f"num_prompts: {num_prompts}")
    print(f"num_warmups: {num_warmups}")
    print(f"goodput_slo passed to vllm bench serve: {goodput_slo}")
    print("CSV goodput definition: request_throughput = completed requests / benchmark duration.")
    print(f"server configs: {len(server_configs)}")
    print(f"arrival rates per server: {arrival_rates}")
    print(f"total benchmark rows: {len(server_configs) * len(arrival_rates)}")
    print()

    for server_idx, server_config in enumerate(server_configs, start=1):
        server_command = _build_server_command(server_config, max_model_len)
        print(
            f"[server {server_idx}/{len(server_configs)}] "
            f"B={server_config.max_num_seqs}, "
            f"gpu_memory_utilization={server_config.gpu_memory_utilization}"
        )
        print("SERVER:", " ".join(server_command))
        for arrival_rate in arrival_rates:
            result_filename = (
                f"vllm_rate_{arrival_rate:g}_B_{server_config.max_num_seqs}_"
                f"gpu_{server_config.gpu_memory_utilization:g}.json"
            )
            bench_command = _build_bench_command(
                server_config=server_config,
                arrival_rate=arrival_rate,
                result_filename=result_filename,
                num_prompts=num_prompts,
                num_warmups=num_warmups,
                goodput_slo=goodput_slo,
                max_model_len=max_model_len,
            )
            print(f"BENCH lambda={arrival_rate:g}:", " ".join(bench_command))
        print("TEARDOWN: terminate vLLM server")
        print()


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
                recent_logs = "\n".join(log_lines[-100:])
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
    recent_logs = "\n".join(log_lines[-160:])
    raise TimeoutError(
        f"vLLM did not become ready after {STARTUP_TIMEOUT_S}s\n"
        f"last vLLM logs:\n{recent_logs}"
    )


def _parse_kv_cache_info(log_text: str) -> dict[str, int | None]:
    block_size = KV_BLOCK_SIZE
    gpu_blocks = None
    gpu_tokens = None

    block_size_match = re.search(r"(?:block_size|block size)\D+(\d+)", log_text, re.I)
    if block_size_match:
        block_size = int(block_size_match.group(1))

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


def _parse_preemption_count(log_text: str) -> int:
    metric_matches = re.findall(
        r"(?:total_)?cumulative_preemption(?:_cnt|_count)?\D+(\d+)",
        log_text,
        flags=re.I,
    )
    if metric_matches:
        return max(int(value) for value in metric_matches)
    return sum(1 for line in log_text.splitlines() if "preempt" in line.lower())


def _write_custom_dataset(records: list[dict[str, Any]], path: Path, num_prompts: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in records[:num_prompts]:
            fh.write(
                json.dumps(
                    {
                        "prompt": row["prompt_text"],
                        "output_tokens": int(row["gen_len"]),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )


def _percentile_value(result: dict[str, Any], metric: str, percentile: float) -> float:
    key = f"p{int(percentile)}_{metric}_ms"
    if key in result:
        return float(result[key]) / 1000.0

    list_key = f"percentiles_{metric}_ms"
    for item in result.get(list_key, []) or []:
        if isinstance(item, dict):
            value = item.get(str(percentile), item.get(percentile))
            if value is not None:
                return float(value) / 1000.0
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            if abs(float(item[0]) - percentile) < 1e-9:
                return float(item[1]) / 1000.0

    if percentile == 50 and f"median_{metric}_ms" in result:
        return float(result[f"median_{metric}_ms"]) / 1000.0
    return 0.0


def _metric_seconds(result: dict[str, Any], key: str) -> float:
    return float(result.get(key, 0.0)) / 1000.0


def _load_benchmark_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_server_command(config: ServerConfig, max_model_len: int) -> list[str]:
    return [
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
        str(max_model_len),
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--max-num-batched-tokens",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
    ]


def _build_bench_command(
    server_config: ServerConfig,
    arrival_rate: float,
    result_filename: str,
    num_prompts: int,
    num_warmups: int,
    goodput_slo: list[str],
    max_model_len: int,
) -> list[str]:
    command = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--host",
        "127.0.0.1",
        "--port",
        str(SERVER_PORT),
        "--endpoint",
        "/v1/completions",
        "--model",
        MODEL_ID,
        "--served-model-name",
        MODEL_ID,
        "--tokenizer",
        MODEL_ID,
        "--dataset-name",
        "custom",
        "--dataset-path",
        REMOTE_DATASET_PATH,
        "--custom-output-len",
        "-1",
        "--num-prompts",
        str(num_prompts),
        "--num-warmups",
        str(num_warmups),
        "--request-rate",
        str(arrival_rate),
        "--burstiness",
        "1.0",
        "--disable-shuffle",
        "--ignore-eos",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        REMOTE_RESULTS_DIR,
        "--result-filename",
        result_filename,
        "--percentile-metrics",
        "ttft,tpot",
        "--metric-percentiles",
        "50,95,99",
        "--metadata",
        f"vllm_version={VLLM_VERSION}",
        f"model={MODEL_ID}",
        f"arrival_rate={arrival_rate}",
        f"max_num_seqs={server_config.max_num_seqs}",
        f"gpu_memory_utilization={server_config.gpu_memory_utilization}",
        f"max_model_len={max_model_len}",
        f"max_num_batched_tokens={max_model_len}",
        f"num_warmups={num_warmups}",
        "csv_goodput_definition=completed_requests_per_benchmark_second",
    ]
    if goodput_slo:
        command.append("--goodput")
        command.extend(goodput_slo)
    return command


def _parse_bench_result(
    *,
    result: dict[str, Any],
    arrival_rate: float,
    server_config: ServerConfig,
    kv_budget: int,
    num_prompts: int,
) -> dict[str, Any]:
    completed = int(result.get("completed", 0))
    failed = int(result.get("failed", max(num_prompts - completed, 0)))
    total = max(completed + failed, num_prompts)
    request_throughput = float(result.get("request_throughput", 0.0))

    return {
        "arrival_rate": arrival_rate,
        "max_batch_size": server_config.max_num_seqs,
        "kv_budget": kv_budget,
        "n_seeds": 1,
        "mean_ttft_mean": _metric_seconds(result, "mean_ttft_ms"),
        "mean_ttft_std": 0.0,
        "p50_ttft_mean": _percentile_value(result, "ttft", 50),
        "p50_ttft_std": 0.0,
        "p95_ttft_mean": _percentile_value(result, "ttft", 95),
        "p95_ttft_std": 0.0,
        "p99_ttft_mean": _percentile_value(result, "ttft", 99),
        "p99_ttft_std": 0.0,
        "goodput_mean": request_throughput,
        "goodput_std": 0.0,
        "blocking_probability_mean": failed / total if total else 0.0,
        "blocking_probability_std": 0.0,
    }


def _run_server_config(
    server_config: ServerConfig,
    arrival_rates: list[float],
    num_prompts: int,
    num_warmups: int,
    goodput_slo: list[str],
    max_model_len: int,
) -> dict[str, Any]:
    base_url = f"http://127.0.0.1:{SERVER_PORT}"
    server_command = _build_server_command(server_config, max_model_len)

    print("starting server:", " ".join(server_command), flush=True)
    proc = subprocess.Popen(
        server_command,
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

    wall_start = time.perf_counter()
    try:
        _wait_for_server(base_url, proc, log_lines)
        print("server ready", flush=True)
        log_text = "\n".join(log_lines)
        kv_info = _parse_kv_cache_info(log_text)
        preemption_count = _parse_preemption_count(log_text)
        kv_budget = (
            kv_info["gpu_kv_cache_tokens"]
            or (
                kv_info["gpu_kv_cache_blocks"] * kv_info["kv_block_size"]
                if kv_info["gpu_kv_cache_blocks"] and kv_info["kv_block_size"]
                else 0
            )
        )
        summary_rows = []
        metadata_rows = []
        for arrival_rate in arrival_rates:
            result_filename = (
                f"vllm_rate_{arrival_rate:g}_B_{server_config.max_num_seqs}_"
                f"gpu_{server_config.gpu_memory_utilization:g}.json"
            )
            result_path = Path(REMOTE_RESULTS_DIR) / result_filename
            bench_command = _build_bench_command(
                server_config=server_config,
                arrival_rate=arrival_rate,
                result_filename=result_filename,
                num_prompts=num_prompts,
                num_warmups=num_warmups,
                goodput_slo=goodput_slo,
                max_model_len=max_model_len,
            )
            print("running bench:", " ".join(bench_command), flush=True)
            bench = subprocess.run(
                bench_command,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            print(bench.stdout, flush=True)
            if bench.returncode != 0:
                raise RuntimeError(
                    f"vllm bench serve failed with code {bench.returncode}\n{bench.stdout}"
                )
            if not result_path.exists():
                raise FileNotFoundError(f"benchmark did not write {result_path}")
            result = _load_benchmark_json(result_path)
            summary_rows.append(
                _parse_bench_result(
                    result=result,
                    arrival_rate=arrival_rate,
                    server_config=server_config,
                    kv_budget=kv_budget,
                    num_prompts=num_prompts,
                )
            )
            completed = int(result.get("completed", 0))
            failed = int(result.get("failed", max(num_prompts - completed, 0)))
            metadata_rows.append(
                {
                    "arrival_rate": arrival_rate,
                    "bench_flags": bench_command,
                    "completed": completed,
                    "failed": failed,
                    "request_throughput": result.get("request_throughput"),
                    "request_goodput_slo_filtered": result.get("request_goodput"),
                    "output_throughput": result.get("output_throughput"),
                    "total_token_throughput": result.get("total_token_throughput"),
                    "mean_tpot_s": _metric_seconds(result, "mean_tpot_ms"),
                    "p99_tpot_s": _percentile_value(result, "tpot", 99),
                    "remote_result_path": str(result_path),
                }
            )

        metadata = {
            "max_num_seqs": server_config.max_num_seqs,
            "gpu_memory_utilization": server_config.gpu_memory_utilization,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_model_len,
            "num_prompts": num_prompts,
            "num_warmups": num_warmups,
            "model": MODEL_ID,
            "vllm_version": VLLM_VERSION,
            "installed_vllm_version": version("vllm"),
            "server_flags": server_command,
            "goodput_slo": goodput_slo,
            "csv_goodput_definition": "completed_requests_per_benchmark_second",
            "kv_cache": kv_info,
            "preemption_count": preemption_count,
            "bench_runs": metadata_rows,
            "total_wall_s": time.perf_counter() - wall_start,
        }
        return {"summary_rows": summary_rows, "metadata": metadata}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)
        print(f"vLLM process return code: {proc.returncode}", flush=True)


@app.function(
    gpu="A10G",
    timeout=43200,
    scaledown_window=20,
    volumes={
        "/root/.cache/vllm": vllm_cache,
        "/root/.cache/huggingface": hf_cache,
    },
)
def run_vllm_sweep_remote(
    server_configs_payload: list[dict[str, Any]],
    arrival_rates: list[float],
    records: list[dict[str, Any]],
    num_prompts: int,
    num_warmups: int,
    goodput_slo: list[str],
    max_model_len: int,
) -> dict[str, Any]:
    Path(REMOTE_WORKDIR).mkdir(parents=True, exist_ok=True)
    Path(REMOTE_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    _write_custom_dataset(records, Path(REMOTE_DATASET_PATH), len(records))
    print(f"wrote custom dataset: {REMOTE_DATASET_PATH}", flush=True)
    print(f"pinned vLLM version: {VLLM_VERSION}", flush=True)
    print(f"installed vLLM version: {version('vllm')}", flush=True)

    summary_rows = []
    metadata_rows = []
    for config_payload in server_configs_payload:
        server_config = ServerConfig(**config_payload)
        result = _run_server_config(
            server_config=server_config,
            arrival_rates=arrival_rates,
            num_prompts=num_prompts,
            num_warmups=num_warmups,
            goodput_slo=goodput_slo,
            max_model_len=max_model_len,
        )
        summary_rows.extend(result["summary_rows"])
        metadata_rows.append(result["metadata"])

    return {
        "summary_rows": summary_rows,
        "metadata": {
            "model": MODEL_ID,
            "vllm_version": VLLM_VERSION,
            "installed_vllm_version": version("vllm"),
            "dataset_path": str(REMOTE_DATASET_PATH),
            "result_dir": str(REMOTE_RESULTS_DIR),
            "goodput_parity_note": (
                "CSV goodput_mean uses vLLM request_throughput, matching SimPy "
                "completed requests divided by experiment/benchmark duration. "
                "vLLM's SLO-filtered request_goodput is stored only in metadata."
            ),
            "volumes": {
                "vllm_cache": "/root/.cache/vllm",
                "huggingface_cache": "/root/.cache/huggingface",
            },
            "configs": metadata_rows,
        },
    }


def run_local(
    trace: str = "data/sharegpt_trace.parquet",
    output: str = "results/vllm_sweep_summary.csv",
    metadata_output: str = "results/vllm_sweep_metadata.json",
    arrival_rates: str = DEFAULT_ARRIVAL_RATES,
    batch_sizes: str = DEFAULT_BATCH_SIZES,
    gpu_memory_utils: str = DEFAULT_GPU_MEMORY_UTILS,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    num_prompts: int = DEFAULT_NUM_PROMPTS,
    num_warmups: int = DEFAULT_NUM_WARMUPS,
    goodput_slo: str = DEFAULT_GOODPUT_SLO,
    smoke: bool = False,
    dry_run: bool = False,
) -> None:
    if smoke:
        num_prompts = min(num_prompts, SMOKE_NUM_PROMPTS)
        num_warmups = min(num_warmups, SMOKE_NUM_WARMUPS)

    arrival_rate_values, server_configs = build_plan(
        arrival_rates=arrival_rates,
        batch_sizes=batch_sizes,
        gpu_memory_utils=gpu_memory_utils,
        smoke=smoke,
    )
    goodput_values = parse_goodput_slo(goodput_slo)

    if dry_run:
        print_dry_run(
            trace=Path(trace),
            output=Path(output),
            metadata_output=Path(metadata_output),
            arrival_rates=arrival_rate_values,
            server_configs=server_configs,
            max_model_len=max_model_len,
            num_prompts=num_prompts,
            num_warmups=num_warmups,
            goodput_slo=goodput_values,
        )
        return

    records = load_trace_records(Path(trace), num_prompts + num_warmups)
    configs_payload = [config.__dict__ for config in server_configs]

    print(f"running {len(server_configs)} vLLM server config(s) on Modal")
    print(f"arrival rates per server: {arrival_rate_values}")
    print(f"num_prompts: {num_prompts}")
    print(f"num_warmups: {num_warmups}")
    print(f"goodput SLO: {goodput_values}")
    result = run_vllm_sweep_remote.remote(
        server_configs_payload=configs_payload,
        arrival_rates=arrival_rate_values,
        records=records,
        num_prompts=num_prompts,
        num_warmups=num_warmups,
        goodput_slo=goodput_values,
        max_model_len=max_model_len,
    )

    write_summary_csv(Path(output), result["summary_rows"])
    write_metadata(Path(metadata_output), result["metadata"])
    print(f"wrote {output}")
    print(f"wrote {metadata_output}")


@app.local_entrypoint()
def main(
    trace: str = "data/sharegpt_trace.parquet",
    output: str = "results/vllm_sweep_summary.csv",
    metadata_output: str = "results/vllm_sweep_metadata.json",
    arrival_rates: str = DEFAULT_ARRIVAL_RATES,
    batch_sizes: str = DEFAULT_BATCH_SIZES,
    gpu_memory_utils: str = DEFAULT_GPU_MEMORY_UTILS,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    num_prompts: int = DEFAULT_NUM_PROMPTS,
    num_warmups: int = DEFAULT_NUM_WARMUPS,
    goodput_slo: str = DEFAULT_GOODPUT_SLO,
    smoke: bool = False,
    dry_run: bool = False,
) -> None:
    run_local(
        trace=trace,
        output=output,
        metadata_output=metadata_output,
        arrival_rates=arrival_rates,
        batch_sizes=batch_sizes,
        gpu_memory_utils=gpu_memory_utils,
        max_model_len=max_model_len,
        num_prompts=num_prompts,
        num_warmups=num_warmups,
        goodput_slo=goodput_slo,
        smoke=smoke,
        dry_run=dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or dry-run the Modal/vLLM ShareGPT serving sweep."
    )
    parser.add_argument("--trace", default="data/sharegpt_trace.parquet")
    parser.add_argument("--output", default="results/vllm_sweep_summary.csv")
    parser.add_argument(
        "--metadata-output", default="results/vllm_sweep_metadata.json"
    )
    parser.add_argument("--arrival-rates", default=DEFAULT_ARRIVAL_RATES)
    parser.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--gpu-memory-utils", default=DEFAULT_GPU_MEMORY_UTILS)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS)
    parser.add_argument("--num-warmups", type=int, default=DEFAULT_NUM_WARMUPS)
    parser.add_argument("--goodput-slo", default=DEFAULT_GOODPUT_SLO)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the full grid and exact commands without invoking Modal.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_local(
        trace=args.trace,
        output=args.output,
        metadata_output=args.metadata_output,
        arrival_rates=args.arrival_rates,
        batch_sizes=args.batch_sizes,
        gpu_memory_utils=args.gpu_memory_utils,
        max_model_len=args.max_model_len,
        num_prompts=args.num_prompts,
        num_warmups=args.num_warmups,
        goodput_slo=args.goodput_slo,
        smoke=args.smoke,
        dry_run=args.dry_run,
    )
