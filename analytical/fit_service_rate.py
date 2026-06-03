from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from simulator.simpy_simulator import iteration_latency as default_iteration_latency


class LatencyModel(Protocol):
    def iteration_latency(self, batch_size: int, tokens: int, phase: str) -> float: ...


@dataclass
class MeasuredLatencyCurve:
    """Measured decode-step curve from Modal/vLLM microbenchmarks.

    The microbenchmark directly fits decode iteration latency as
    base + scale * (batch - 1) ** alpha. It does not benchmark prefill, so
    prefill stays on the existing simulator placeholder unless a future
    latency JSON adds explicit prefill coefficients.
    """

    base_s: float
    scale_s: float
    alpha: float
    prefill_base_s: float | None = None
    prefill_per_token_s: float | None = None

    def decode_latency(self, batch_size: int) -> float:
        batch = max(1, batch_size)
        return self.base_s + self.scale_s * (batch - 1) ** self.alpha

    def iteration_latency(self, batch_size: int, tokens: int, phase: str) -> float:
        if phase == "decode":
            return self.decode_latency(batch_size)
        if self.prefill_base_s is not None and self.prefill_per_token_s is not None:
            batch = max(1, batch_size)
            return self.prefill_base_s + self.prefill_per_token_s * tokens * batch
        return default_iteration_latency(batch_size, tokens, phase)


def load_latency_curve(path: Path | str | None) -> MeasuredLatencyCurve | None:
    if path is None:
        return None
    with Path(path).open(encoding="utf-8") as fh:
        payload = json.load(fh)
    fit = payload["fit"]
    return MeasuredLatencyCurve(
        base_s=float(fit["base_s"]),
        scale_s=float(fit["scale_s"]),
        alpha=float(fit["alpha"]),
        prefill_base_s=fit.get("prefill_base_s"),
        prefill_per_token_s=fit.get("prefill_per_token_s"),
    )


def get_iteration_latency(
    latency_model: LatencyModel | None, batch_size: int, tokens: int, phase: str
) -> float:
    if latency_model is None:
        return default_iteration_latency(batch_size, tokens, phase)
    return latency_model.iteration_latency(batch_size, tokens, phase)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a measured latency curve.")
    parser.add_argument(
        "--latency-curve",
        type=Path,
        default=Path("results") / "latency_curve.json",
    )
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_latency_curve(args.latency_curve)
    if model is None:
        raise FileNotFoundError(args.latency_curve)
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item]
    print(f"loaded {args.latency_curve}")
    print({"base_s": model.base_s, "scale_s": model.scale_s, "alpha": model.alpha})
    for batch in batch_sizes:
        print(f"batch={batch}: decode_step={model.decode_latency(batch):.6f}s")


if __name__ == "__main__":
    main()
