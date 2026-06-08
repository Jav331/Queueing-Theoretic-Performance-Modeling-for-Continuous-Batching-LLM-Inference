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
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

POLICY_STYLE = {
    "fcfs": {"color": "#4C78A8", "marker": "o", "label": "FCFS"},
    "srpt": {"color": "#F58518", "marker": "s", "label": "SRPT"},
    "dp": {"color": "#54A24B", "marker": "^", "label": "DP"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot FCFS/SRPT/DP comparison from policy_comparison.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results") / "policy_comparison.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures") / "policy_comparison",
    )
    return parser.parse_args()


def plot_metric_vs_arrival_per_cell(
    df: pd.DataFrame, metric: str, ylabel: str, output_dir: Path
) -> None:
    for (max_batch_size, kv_budget), cell in df.groupby(["max_batch_size", "kv_budget"]):
        plt.figure(figsize=(8, 5))
        for policy_name, group in cell.groupby("policy"):
            style = POLICY_STYLE.get(policy_name, {})
            grouped = group.sort_values("arrival_rate")
            plt.plot(
                grouped["arrival_rate"],
                grouped[f"{metric}_mean"],
                marker=style.get("marker", "o"),
                color=style.get("color"),
                linewidth=2,
                label=style.get("label", policy_name),
            )
            std_col = f"{metric}_std"
            if std_col in grouped:
                lower = grouped[f"{metric}_mean"] - grouped[std_col]
                upper = grouped[f"{metric}_mean"] + grouped[std_col]
                plt.fill_between(
                    grouped["arrival_rate"], lower, upper, alpha=0.15, color=style.get("color")
                )
        plt.xlabel("Arrival rate (req/s)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs arrival rate (B={max_batch_size}, K={kv_budget})")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Policy")
        plt.tight_layout()
        path = output_dir / f"{metric}_B{max_batch_size}_K{kv_budget}.png"
        plt.savefig(path, dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} missing. Run `python -m experiments.run_policy_comparison` first."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    plot_metric_vs_arrival_per_cell(df, "p99_ttft", "p99 TTFT (s)", args.output_dir)
    plot_metric_vs_arrival_per_cell(df, "goodput", "Goodput (req/s)", args.output_dir)
    plot_metric_vs_arrival_per_cell(
        df, "blocking_probability", "Blocking probability", args.output_dir
    )
    cells = df.groupby(["max_batch_size", "kv_budget"]).size().shape[0]
    print(f"wrote {3 * cells} figures under {args.output_dir}")


if __name__ == "__main__":
    main()
