from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
from statistics import mean, pstdev

from tqdm import tqdm

from simulator.metrics import summarize
from simulator.simpy_simulator import ContinuousBatchingSimulator


DEFAULT_ARRIVAL_RATES = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
DEFAULT_MAX_BATCH_SIZES = [1, 2, 4, 8, 16]
DEFAULT_KV_BUDGETS = [2048, 4096, 8192, 16384]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
METRIC_COLUMNS = [
    "mean_ttft",
    "p50_ttft",
    "p95_ttft",
    "p99_ttft",
    "goodput",
    "blocking_probability",
]

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


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a baseline simulator sweep.")
    parser.add_argument("--duration", type=float, default=1000.0)
    parser.add_argument("--prompt-mean", type=float, default=256.0)
    parser.add_argument("--generation-mean", type=float, default=128.0)
    parser.add_argument(
        "--arrival-rates",
        type=parse_float_list,
        default=DEFAULT_ARRIVAL_RATES,
        help="Comma-separated arrival rates. Default: 0.5,1,2,3,4,5,6,8",
    )
    parser.add_argument(
        "--max-batch-sizes",
        type=parse_int_list,
        default=DEFAULT_MAX_BATCH_SIZES,
        help="Comma-separated max batch sizes. Default: 1,2,4,8,16",
    )
    parser.add_argument(
        "--kv-budgets",
        type=parse_int_list,
        default=DEFAULT_KV_BUDGETS,
        help="Comma-separated KV budgets. Default: 2048,4096,8192,16384",
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=DEFAULT_SEEDS,
        help="Comma-separated random seeds. Default: 0,1,2,3,4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "sweep_summary.csv",
        help="CSV path for sweep-level summary metrics.",
    )
    return parser.parse_args()


def run_one_config(
    *,
    duration: float,
    arrival_rate: float,
    max_batch_size: int,
    kv_budget: int,
    seed: int,
    prompt_mean: float,
    generation_mean: float,
) -> dict[str, float]:
    simulator = ContinuousBatchingSimulator(
        duration=duration,
        arrival_rate=arrival_rate,
        max_batch_size=max_batch_size,
        kv_budget=kv_budget,
        seed=seed,
        prompt_mean=prompt_mean,
        generation_mean=generation_mean,
    )
    summary = summarize(simulator.run(), duration)
    return {
        "arrival_rate": arrival_rate,
        "max_batch_size": max_batch_size,
        "kv_budget": kv_budget,
        "mean_ttft": summary["mean_ttft"],
        "p50_ttft": summary["p50_ttft"],
        "p95_ttft": summary["p95_ttft"],
        "p99_ttft": summary["p99_ttft"],
        "goodput": summary["goodput_rps"],
        "blocking_probability": summary["blocking_probability"],
    }


def aggregate_config(seed_rows: list[dict[str, float]]) -> dict[str, float]:
    first = seed_rows[0]
    row = {
        "arrival_rate": first["arrival_rate"],
        "max_batch_size": first["max_batch_size"],
        "kv_budget": first["kv_budget"],
        "n_seeds": len(seed_rows),
    }
    for metric in METRIC_COLUMNS:
        values = [seed_row[metric] for seed_row in seed_rows]
        row[f"{metric}_mean"] = mean(values)
        row[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return row


def main() -> None:
    args = parse_args()
    configs = list(
        product(args.arrival_rates, args.max_batch_sizes, args.kv_budgets)
    )
    rows = []

    for arrival_rate, max_batch_size, kv_budget in tqdm(configs, desc="sweep configs"):
        seed_rows = []
        for seed in args.seeds:
            seed_rows.append(
                run_one_config(
                    duration=args.duration,
                    arrival_rate=arrival_rate,
                    max_batch_size=max_batch_size,
                    kv_budget=kv_budget,
                    seed=seed,
                    prompt_mean=args.prompt_mean,
                    generation_mean=args.generation_mean,
                )
            )
        rows.append(aggregate_config(seed_rows))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.output}")
    print(f"configs: {len(rows)}")
    print(f"seeds per config: {len(args.seeds)}")


if __name__ == "__main__":
    main()
