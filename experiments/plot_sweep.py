from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot baseline sweep results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results") / "sweep_summary.csv",
        help="Sweep summary CSV produced by experiments.run_sweep.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory where PNG figures should be written.",
    )
    parser.add_argument(
        "--fixed-kv-budget",
        type=int,
        default=8192,
        help="KV budget to use for arrival-rate plots.",
    )
    parser.add_argument(
        "--fixed-batch-size",
        type=int,
        default=8,
        help="Max batch size to use for KV-budget plots.",
    )
    return parser.parse_args()


def require_fixed_value(df: pd.DataFrame, column: str, requested: int) -> int:
    values = sorted(df[column].unique())
    if requested in values:
        return requested
    raise ValueError(
        f"Requested {column}={requested}, but the CSV only contains {values}."
    )


def plot_by_batch(
    df: pd.DataFrame,
    *,
    y_column: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for max_batch_size, group in sorted(df.groupby("max_batch_size")):
        grouped = group.sort_values("arrival_rate")
        plt.plot(
            grouped["arrival_rate"],
            grouped[f"{y_column}_mean"],
            marker="o",
            linewidth=2,
            label=f"B={max_batch_size}",
        )
        std_column = f"{y_column}_std"
        if std_column in grouped:
            lower = grouped[f"{y_column}_mean"] - grouped[std_column]
            upper = grouped[f"{y_column}_mean"] + grouped[std_column]
            plt.fill_between(grouped["arrival_rate"], lower, upper, alpha=0.12)

    plt.xlabel("Arrival rate (requests/s)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Max batch size")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_by_kv(
    df: pd.DataFrame,
    *,
    y_column: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for arrival_rate, group in sorted(df.groupby("arrival_rate")):
        grouped = group.sort_values("kv_budget")
        plt.plot(
            grouped["kv_budget"],
            grouped[f"{y_column}_mean"],
            marker="o",
            linewidth=2,
            label=f"λ={arrival_rate:g} req/s",
        )
        std_column = f"{y_column}_std"
        if std_column in grouped:
            lower = grouped[f"{y_column}_mean"] - grouped[std_column]
            upper = grouped[f"{y_column}_mean"] + grouped[std_column]
            plt.fill_between(grouped["kv_budget"], lower, upper, alpha=0.12)

    plt.xlabel("KV Budget (tokens)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Arrival rate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} does not exist. Run `python -m experiments.run_sweep` first."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    fixed_kv_budget = require_fixed_value(df, "kv_budget", args.fixed_kv_budget)
    fixed_batch_size = require_fixed_value(
        df, "max_batch_size", args.fixed_batch_size
    )
    arrival_df = df[df["kv_budget"] == fixed_kv_budget]
    kv_df = df[df["max_batch_size"] == fixed_batch_size]

    plot_by_batch(
        arrival_df,
        y_column="p99_ttft",
        y_label="p99 TTFT (s)",
        title=f"p99 TTFT vs Arrival Rate (KV={fixed_kv_budget})",
        output_path=args.output_dir / "p99_ttft_vs_arrival_rate.png",
    )
    plot_by_batch(
        arrival_df,
        y_column="goodput",
        y_label="Goodput (completed requests/s)",
        title=f"Goodput vs Arrival Rate (KV={fixed_kv_budget})",
        output_path=args.output_dir / "goodput_vs_arrival_rate.png",
    )
    plot_by_kv(
        kv_df,
        y_column="blocking_probability",
        y_label="Blocking Probability",
        title=f"Blocking Probability vs KV Budget (B={fixed_batch_size})",
        output_path=args.output_dir / "blocking_probability_vs_kv_budget.png",
    )
    plot_by_kv(
        kv_df,
        y_column="p99_ttft",
        y_label="p99 TTFT (s)",
        title=f"p99 TTFT vs KV Budget (B={fixed_batch_size})",
        output_path=args.output_dir / "p99_ttft_vs_kv_budget.png",
    )

    print(f"arrival-rate plots use KV={fixed_kv_budget}")
    print(f"KV-budget plots use B={fixed_batch_size}")
    print(f"wrote {args.output_dir / 'p99_ttft_vs_arrival_rate.png'}")
    print(f"wrote {args.output_dir / 'goodput_vs_arrival_rate.png'}")
    print(f"wrote {args.output_dir / 'blocking_probability_vs_kv_budget.png'}")
    print(f"wrote {args.output_dir / 'p99_ttft_vs_kv_budget.png'}")


if __name__ == "__main__":
    main()
