from __future__ import annotations

import argparse
import csv
from pathlib import Path

from simulator.metrics import records_as_rows, summarize
from simulator.simpy_simulator import ContinuousBatchingSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the baseline SimPy simulator.")
    parser.add_argument("--duration", type=float, default=200.0)
    parser.add_argument("--arrival-rate", type=float, default=2.0)
    parser.add_argument("--max-batch-size", "-B", type=int, default=8)
    parser.add_argument("--kv-budget", "-K", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prompt-mean", type=float, default=256.0)
    parser.add_argument("--generation-mean", type=float, default=128.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "baseline.csv",
        help="CSV path for per-request results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulator = ContinuousBatchingSimulator(
        duration=args.duration,
        arrival_rate=args.arrival_rate,
        max_batch_size=args.max_batch_size,
        kv_budget=args.kv_budget,
        seed=args.seed,
        prompt_mean=args.prompt_mean,
        generation_mean=args.generation_mean,
    )
    records = simulator.run()
    rows = records_as_rows(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summary = summarize(records, args.duration)
    print(f"wrote {args.output}")
    print(f"requests: {summary['total_requests']}")
    print(f"accepted: {summary['accepted']}")
    print(f"rejected: {summary['rejected']}")
    print(f"completed: {summary['completed']}")
    print(f"blocking probability: {summary['blocking_probability']:.4f}")
    print(f"goodput: {summary['goodput_rps']:.4f} completed requests/s")
    print(f"output throughput: {summary['output_tokens_per_s']:.2f} tokens/s")
    print(f"mean TTFT: {summary['mean_ttft']:.4f}s")
    print(f"p50 TTFT: {summary['p50_ttft']:.4f}s")
    print(f"p95 TTFT: {summary['p95_ttft']:.4f}s")
    print(f"p99 TTFT: {summary['p99_ttft']:.4f}s")


if __name__ == "__main__":
    main()
