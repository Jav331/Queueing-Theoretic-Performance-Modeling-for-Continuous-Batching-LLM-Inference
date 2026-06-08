from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = {
    "mean_ttft": {
        "label": "Mean TTFT (s)",
        "title": "Simulation vs Analytical Mean TTFT",
        "output": "comparison_mean_ttft.png",
    },
    "p99_ttft": {
        "label": "p99 TTFT (s)",
        "title": "Simulation vs Analytical p99 TTFT",
        "output": "comparison_p99_ttft.png",
    },
    "goodput": {
        "label": "Goodput (completed requests/s)",
        "title": "Simulation vs Analytical Goodput",
        "output": "comparison_goodput.png",
    },
    "blocking_probability": {
        "label": "Blocking Probability",
        "title": "Simulation vs Analytical Blocking Probability",
        "output": "comparison_blocking_probability.png",
    },
}

plt.rcParams.update(
    {
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot simulator vs analytical comparison results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results") / "comparison_summary.csv",
        help="Comparison CSV produced by experiments.compare_simulation_analytical.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory where PNG figures should be written.",
    )
    return parser.parse_args()


def plot_scatter(df: pd.DataFrame, metric: str, output_dir: Path) -> Path:
    config = METRICS[metric]
    sim_column = f"{metric}_sim"
    analytical_column = f"{metric}_analytical"

    x = df[sim_column]
    y = df[analytical_column]
    upper = max(x.max(), y.max())
    padding = 0.05 * upper if upper > 0 else 0.05
    axis_max = upper + padding

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.75, edgecolors="none")
    plt.plot(
        [0.0, axis_max],
        [0.0, axis_max],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="y=x",
    )
    plt.xlim(0.0, axis_max)
    plt.ylim(0.0, axis_max)
    plt.xlabel(f"Simulation {config['label']}")
    plt.ylabel(f"Analytical {config['label']}")
    plt.title(config["title"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / config["output"]
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_relative_errors(df: pd.DataFrame, output_dir: Path) -> Path:
    labels = []
    means = []
    for metric, config in METRICS.items():
        labels.append(config["label"])
        means.append(df[f"{metric}_relative_error"].mean())

    plt.figure(figsize=(9, 5))
    plt.bar(labels, means, color="#4C78A8")
    plt.ylabel("Mean Relative Error")
    plt.title("Mean Relative Error by Metric")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "comparison_relative_errors.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} does not exist. Run `python -m experiments.compare_simulation_analytical` first."
        )

    df = pd.read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        output_path = plot_scatter(df, metric, args.output_dir)
        print(f"wrote {output_path}")
    print(f"wrote {plot_relative_errors(df, args.output_dir)}")


if __name__ == "__main__":
    main()
