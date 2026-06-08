from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
from statistics import mean, pstdev

from tqdm import tqdm

from experiments.run_sweep import parse_float_list, parse_int_list
from simulator.metrics import summarize
from simulator.policies import DPTablePolicy, FCFSPolicy, SRPTPolicy
from simulator.policy_simulator import PolicySimulator


DEFAULT_ARRIVAL_RATES = [1.0, 2.0, 3.0, 4.0]
DEFAULT_MAX_BATCH_SIZES = [4, 8]
DEFAULT_KV_BUDGETS = [4096, 8192, 16384]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_POLICIES = ["fcfs", "srpt", "dp"]

METRIC_COLUMNS = [
    "mean_ttft",
    "p50_ttft",
    "p95_ttft",
    "p99_ttft",
    "goodput",
    "blocking_probability",
]
SUMMARY_COLUMNS = [
    "policy",
    "arrival_rate",
    "max_batch_size",
    "kv_budget",
    "n_seeds",
]
for metric in METRIC_COLUMNS:
    SUMMARY_COLUMNS.extend([f"{metric}_mean", f"{metric}_std"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep FCFS/SRPT/DP admission policies and summarize."
    )
    parser.add_argument("--duration", type=float, default=500.0)
    parser.add_argument("--prompt-mean", type=float, default=256.0)
    parser.add_argument("--generation-mean", type=float, default=128.0)
    parser.add_argument("--arrival-rates", type=parse_float_list, default=DEFAULT_ARRIVAL_RATES)
    parser.add_argument("--max-batch-sizes", type=parse_int_list, default=DEFAULT_MAX_BATCH_SIZES)
    parser.add_argument("--kv-budgets", type=parse_int_list, default=DEFAULT_KV_BUDGETS)
    parser.add_argument("--seeds", type=parse_int_list, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--policies",
        type=lambda v: [s.strip() for s in v.split(",") if s.strip()],
        default=DEFAULT_POLICIES,
    )
    parser.add_argument(
        "--dp-table-dir",
        type=Path,
        default=Path("results") / "dp_policies",
        help="Directory containing dp_lam<L>_B<B>_K<K>.npy files.",
    )
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "policy_comparison.csv",
    )
    return parser.parse_args()


def make_policy(name: str, *, arrival_rate, max_batch_size, kv_budget, dp_table_dir, page_size):
    if name == "fcfs":
        return FCFSPolicy(), "fcfs"
    if name == "srpt":
        return SRPTPolicy(), "srpt"
    if name == "dp":
        fname = f"dp_lam{arrival_rate:g}_B{max_batch_size}_K{kv_budget}.npy"
        path = dp_table_dir / fname
        if not path.exists():
            return None, None
        return DPTablePolicy(path, page_size=page_size), "fcfs"
    raise ValueError(name)


def run_one(*, duration, arrival_rate, max_batch_size, kv_budget, seed, prompt_mean,
            generation_mean, policy, scheduling):
    sim = PolicySimulator(
        policy=policy,
        scheduling=scheduling,
        duration=duration,
        arrival_rate=arrival_rate,
        max_batch_size=max_batch_size,
        kv_budget=kv_budget,
        seed=seed,
        prompt_mean=prompt_mean,
        generation_mean=generation_mean,
    )
    summary = summarize(sim.run(), duration)
    return {
        "mean_ttft": summary["mean_ttft"],
        "p50_ttft": summary["p50_ttft"],
        "p95_ttft": summary["p95_ttft"],
        "p99_ttft": summary["p99_ttft"],
        "goodput": summary["goodput_rps"],
        "blocking_probability": summary["blocking_probability"],
    }


def aggregate(seed_rows):
    row = {"n_seeds": len(seed_rows)}
    for metric in METRIC_COLUMNS:
        values = [r[metric] for r in seed_rows]
        row[f"{metric}_mean"] = mean(values)
        row[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return row


def main() -> None:
    args = parse_args()
    configs = list(
        product(args.policies, args.arrival_rates, args.max_batch_sizes, args.kv_budgets)
    )
    out_rows = []
    skipped = 0
    for policy_name, arrival_rate, max_batch_size, kv_budget in tqdm(configs, desc="policy sweep"):
        policy, scheduling = make_policy(
            policy_name,
            arrival_rate=arrival_rate,
            max_batch_size=max_batch_size,
            kv_budget=kv_budget,
            dp_table_dir=args.dp_table_dir,
            page_size=args.page_size,
        )
        if policy is None:
            skipped += 1
            continue
        seed_rows = []
        for seed in args.seeds:
            seed_rows.append(
                run_one(
                    duration=args.duration,
                    arrival_rate=arrival_rate,
                    max_batch_size=max_batch_size,
                    kv_budget=kv_budget,
                    seed=seed,
                    prompt_mean=args.prompt_mean,
                    generation_mean=args.generation_mean,
                    policy=policy,
                    scheduling=scheduling,
                )
            )
        row = {
            "policy": policy_name,
            "arrival_rate": arrival_rate,
            "max_batch_size": max_batch_size,
            "kv_budget": kv_budget,
        }
        row.update(aggregate(seed_rows))
        out_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"wrote {args.output}")
    print(f"configs: {len(out_rows)} ({skipped} skipped for missing DP tables)")


if __name__ == "__main__":
    main()
