from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

from analytical.approximation import approximate_metrics
from experiments.run_sweep import parse_float_list, parse_int_list


DEFAULT_ARRIVAL_RATES = [0.5, 1.0, 2.0, 3.0]
DEFAULT_MAX_BATCH_SIZES = [1, 4, 8]
DEFAULT_KV_BUDGETS = [2048, 4096, 8192, 16384]

SUMMARY_COLUMNS = [
    "arrival_rate",
    "max_batch_size",
    "kv_budget",
    "mean_ttft",
    "p50_ttft",
    "p95_ttft",
    "p99_ttft",
    "goodput",
    "blocking_probability",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analytical TTFT approximation.")
    parser.add_argument("--prompt-mean", type=float, default=256.0)
    parser.add_argument("--generation-mean", type=float, default=128.0)
    parser.add_argument(
        "--arrival-rates",
        type=parse_float_list,
        default=DEFAULT_ARRIVAL_RATES,
        help="Comma-separated arrival rates. Default: 0.5,1,2,3",
    )
    parser.add_argument(
        "--max-batch-sizes",
        type=parse_int_list,
        default=DEFAULT_MAX_BATCH_SIZES,
        help="Comma-separated max batch sizes. Default: 1,4,8",
    )
    parser.add_argument(
        "--kv-budgets",
        type=parse_int_list,
        default=DEFAULT_KV_BUDGETS,
        help="Comma-separated KV budgets. Default: 2048,4096,8192,16384",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "analytical_summary.csv",
        help="CSV path for analytical summary metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        approximate_metrics(
            arrival_rate,
            max_batch_size,
            kv_budget,
            prompt_mean=args.prompt_mean,
            generation_mean=args.generation_mean,
        )
        for arrival_rate, max_batch_size, kv_budget in product(
            args.arrival_rates, args.max_batch_sizes, args.kv_budgets
        )
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.output}")
    print(f"configs: {len(rows)}")


if __name__ == "__main__":
    main()
