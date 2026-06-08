from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
from tqdm import tqdm

from analytical.dp_policy import derive_admission_policy
from analytical.state_space import DEFAULT_PAGE_SIZE
from experiments.run_sweep import parse_float_list, parse_int_list


DEFAULT_ARRIVAL_RATES = [1.0, 2.0, 3.0, 4.0]
DEFAULT_MAX_BATCH_SIZES = [4, 8]
DEFAULT_KV_BUDGETS = [4096, 8192, 16384]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive DP admission policy tables over a (lambda, B, K) grid."
    )
    parser.add_argument(
        "--slo",
        type=float,
        required=True,
        help="p99 TTFT SLO in seconds; admission penalized when prediction exceeds this.",
    )
    parser.add_argument("--penalty", type=float, default=1.0)
    parser.add_argument("--prompt-mean", type=float, default=256.0)
    parser.add_argument("--generation-mean", type=float, default=128.0)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--q-bins", type=int, default=32)
    parser.add_argument("--kv-bins", type=int, default=32)
    parser.add_argument("--arrival-rates", type=parse_float_list, default=DEFAULT_ARRIVAL_RATES)
    parser.add_argument("--max-batch-sizes", type=parse_int_list, default=DEFAULT_MAX_BATCH_SIZES)
    parser.add_argument("--kv-budgets", type=parse_int_list, default=DEFAULT_KV_BUDGETS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "dp_policies",
        help="One .npy per (lambda, B, K) cell, named dp_lam{lam}_B{B}_K{K}.npy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = list(
        product(args.arrival_rates, args.max_batch_sizes, args.kv_budgets)
    )
    written = []
    for arrival_rate, max_batch_size, kv_budget in tqdm(configs, desc="dp configs"):
        result = derive_admission_policy(
            arrival_rate,
            max_batch_size,
            kv_budget,
            slo_seconds=args.slo,
            penalty=args.penalty,
            prompt_mean=args.prompt_mean,
            generation_mean=args.generation_mean,
            n_classes=args.n_classes,
            page_size=args.page_size,
            q_bins=args.q_bins,
            kv_bins=args.kv_bins,
        )
        fname = f"dp_lam{arrival_rate:g}_B{max_batch_size}_K{kv_budget}.npy"
        np.save(args.output_dir / fname, result.policy_table)
        written.append(
            (fname, result.n_states, result.blocking_probability, result.expected_reward)
        )

    print(f"wrote {len(written)} policy tables under {args.output_dir}")
    print(f"{'file':50s}  states  block   reward")
    for fname, n_states, block, reward in written:
        print(f"{fname:50s}  {n_states:6d}  {block:.3f}  {reward:.3f}")


if __name__ == "__main__":
    main()
