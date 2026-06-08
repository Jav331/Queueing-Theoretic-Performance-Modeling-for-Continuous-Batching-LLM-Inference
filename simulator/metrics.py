from __future__ import annotations

from dataclasses import asdict
from statistics import mean

import numpy as np


def summarize(records, duration: float) -> dict[str, float]:
    accepted = [r for r in records if r.accepted]
    completed = [r for r in accepted if r.finish_time is not None]
    ttfts = [r.ttft for r in completed if r.ttft is not None]
    latencies = [r.latency for r in completed if r.latency is not None]
    output_tokens = sum(r.generation_len for r in completed)

    summary = {
        "total_requests": len(records),
        "accepted": len(accepted),
        "rejected": len(records) - len(accepted),
        "completed": len(completed),
        "blocking_probability": (len(records) - len(accepted)) / len(records)
        if records
        else 0.0,
        "goodput_rps": len(completed) / duration if duration > 0 else 0.0,
        "output_tokens_per_s": output_tokens / duration if duration > 0 else 0.0,
        "mean_latency": mean(latencies) if latencies else 0.0,
        "mean_ttft": mean(ttfts) if ttfts else 0.0,
        "p50_ttft": float(np.percentile(ttfts, 50)) if ttfts else 0.0,
        "p95_ttft": float(np.percentile(ttfts, 95)) if ttfts else 0.0,
        "p99_ttft": float(np.percentile(ttfts, 99)) if ttfts else 0.0,
    }
    return summary


def records_as_rows(records) -> list[dict]:
    return [asdict(record) for record in records]
