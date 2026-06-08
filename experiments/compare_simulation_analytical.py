from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["arrival_rate", "max_batch_size", "kv_budget"]
METRIC_COLUMNS = [
    "mean_ttft",
    "p95_ttft",
    "p99_ttft",
    "goodput",
    "blocking_probability",
]
ANALYTICAL_COLUMNS = [
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
        description="Compare simulator sweep results against analytical approximation."
    )
    parser.add_argument(
        "--simulation",
        type=Path,
        default=Path("results") / "sweep_summary.csv",
    )
    parser.add_argument(
        "--analytical",
        type=Path,
        default=Path("results") / "analytical_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "comparison_summary.csv",
    )
    return parser.parse_args()


def expected_format() -> None:
    print("Expected analytical_summary.csv columns:")
    print(",".join(ANALYTICAL_COLUMNS))
    print("Generate it with: python -m analytical.run_analytical")


def relative_error(estimated: pd.Series, observed: pd.Series) -> pd.Series:
    denominator = observed.abs()
    return (estimated - observed).abs().where(denominator > 0, 0.0) / denominator.where(
        denominator > 0, 1.0
    )


def main() -> None:
    args = parse_args()
    if not args.analytical.exists():
        print(f"Missing {args.analytical}")
        expected_format()
        return
    if not args.simulation.exists():
        raise FileNotFoundError(
            f"{args.simulation} does not exist. Run `python -m experiments.run_sweep` first."
        )

    sim = pd.read_csv(args.simulation)
    analytical = pd.read_csv(args.analytical)

    missing = [column for column in ANALYTICAL_COLUMNS if column not in analytical]
    if missing:
        print(f"{args.analytical} is missing columns: {missing}")
        expected_format()
        return

    sim_columns = KEY_COLUMNS + [f"{metric}_mean" for metric in METRIC_COLUMNS]
    missing_sim = [column for column in sim_columns if column not in sim]
    if missing_sim:
        raise ValueError(f"{args.simulation} is missing columns: {missing_sim}")

    sim = sim[sim_columns].rename(
        columns={f"{metric}_mean": f"{metric}_sim" for metric in METRIC_COLUMNS}
    )
    analytical = analytical[KEY_COLUMNS + METRIC_COLUMNS].rename(
        columns={metric: f"{metric}_analytical" for metric in METRIC_COLUMNS}
    )

    merged = sim.merge(analytical, on=KEY_COLUMNS, how="inner")
    for metric in METRIC_COLUMNS:
        merged[f"{metric}_relative_error"] = relative_error(
            merged[f"{metric}_analytical"], merged[f"{metric}_sim"]
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    print(f"wrote {args.output}")
    print(f"matched configs: {len(merged)}")
    for metric in METRIC_COLUMNS:
        avg_error = merged[f"{metric}_relative_error"].mean()
        print(f"avg relative error {metric}: {avg_error:.4f}")


if __name__ == "__main__":
    main()
