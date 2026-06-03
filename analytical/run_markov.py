from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

from tqdm import tqdm

from analytical.markov_solver import markov_metrics
from analytical.fit_service_rate import load_latency_curve
from analytical.state_space import DEFAULT_PAGE_SIZE
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
    parser = argparse.ArgumentParser(
        description="Solve the multi-class Markov chain over (n_1, ..., n_K)."
    )
    parser.add_argument("--prompt-mean", type=float, default=256.0)
    parser.add_argument("--generation-mean", type=float, default=128.0)
    parser.add_argument(
        "--n-classes",
        type=int,
        default=2,
        help="Number of request size classes (equal-probability bins of the joint lognormal).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help="KV page size in tokens (default 16; matches vLLM PagedAttention).",
    )
    parser.add_argument(
        "--arrival-rates",
        type=parse_float_list,
        default=DEFAULT_ARRIVAL_RATES,
    )
    parser.add_argument(
        "--max-batch-sizes",
        type=parse_int_list,
        default=DEFAULT_MAX_BATCH_SIZES,
    )
    parser.add_argument(
        "--kv-budgets",
        type=parse_int_list,
        default=DEFAULT_KV_BUDGETS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "markov_summary.csv",
    )
    parser.add_argument(
        "--latency-curve",
        type=Path,
        default=None,
        help="Optional results/latency_curve.json from Modal microbenchmarks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latency_model = load_latency_curve(args.latency_curve)
    configs = list(
        product(args.arrival_rates, args.max_batch_sizes, args.kv_budgets)
    )
    rows = []
    max_states = 0
    for arrival_rate, max_batch_size, kv_budget in tqdm(configs, desc="markov configs"):
        result = markov_metrics(
            arrival_rate,
            max_batch_size,
            kv_budget,
            prompt_mean=args.prompt_mean,
            generation_mean=args.generation_mean,
            n_classes=args.n_classes,
            page_size=args.page_size,
            latency_model=latency_model,
        )
        max_states = max(max_states, result.pop("n_states"))
        rows.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.output}")
    print(f"configs: {len(rows)}")
    print(f"largest state space: {max_states}")


if __name__ == "__main__":
    main()
