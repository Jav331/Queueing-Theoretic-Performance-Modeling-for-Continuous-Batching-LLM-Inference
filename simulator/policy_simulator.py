"""Subclass of Adam's ContinuousBatchingSimulator that consults an
injected admission policy (Phase 3). Does NOT modify simpy_simulator.py.

Behavior preserved:
- Iteration latency curve, prefill/decode mechanics, metrics collection,
  RequestRecord layout, KV bookkeeping — all inherited unchanged.

Behavior overridden:
- `_arrival_process` now delegates accept/reject to the AdmissionPolicy.
- `_fill_batch` honors a `scheduling` parameter ("fcfs" | "srpt"). Under
  "srpt", the waiting deque is sorted by kv_required ascending before each
  fill — matching the SRPT batch-selection semantics in `policies.py`.
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from simulator.metrics import records_as_rows, summarize
from simulator.policies import (
    AdmissionPolicy,
    DPTablePolicy,
    FCFSPolicy,
    SRPTPolicy,
    SimState,
)
from simulator.simpy_simulator import ActiveRequest, ContinuousBatchingSimulator, RequestRecord


class PolicySimulator(ContinuousBatchingSimulator):
    def __init__(
        self,
        *,
        policy: AdmissionPolicy,
        scheduling: str = "fcfs",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.policy = policy
        self.scheduling = scheduling.lower()
        if self.scheduling not in {"fcfs", "srpt"}:
            raise ValueError(f"scheduling must be 'fcfs' or 'srpt'; got {scheduling}")

    def _arrival_process(self):
        request_id = 0
        while self.env.now < self.duration:
            interarrival = self.rng.exponential(1.0 / self.arrival_rate)
            yield self.env.timeout(interarrival)
            if self.env.now > self.duration:
                break

            prompt_len = max(1, int(self.rng.lognormal(np.log(self.prompt_mean), 0.55)))
            generation_len = max(
                1, int(self.rng.lognormal(np.log(self.generation_mean), 0.70))
            )
            kv_required = prompt_len + generation_len
            record = RequestRecord(
                request_id=request_id,
                arrival_time=self.env.now,
                prompt_len=prompt_len,
                generation_len=generation_len,
                kv_required=kv_required,
                accepted=False,
            )
            request_id += 1

            state = SimState(
                reserved_kv=self.reserved_kv,
                kv_budget=self.kv_budget,
                active_count=len(self.active),
                waiting_count=len(self.waiting),
                max_batch_size=self.max_batch_size,
            )
            if self.policy.admit(record, state):
                record.accepted = True
                record.admission_time = self.env.now
                self.reserved_kv += kv_required
                self.waiting.append(ActiveRequest(record=record))
                self._wake_server()
            else:
                record.rejected_time = self.env.now

            self.records.append(record)

    def _fill_batch(self) -> None:
        if self.scheduling == "srpt" and self.waiting:
            self.waiting = deque(
                sorted(self.waiting, key=lambda req: req.record.kv_required)
            )
        super()._fill_batch()


def build_policy(name: str, *, policy_table: Path | None, page_size: int) -> AdmissionPolicy:
    name = name.lower()
    if name == "fcfs":
        return FCFSPolicy()
    if name == "srpt":
        return SRPTPolicy()
    if name == "dp":
        if policy_table is None:
            raise ValueError("DP policy requires --policy-table <path>")
        return DPTablePolicy(policy_table, page_size=page_size)
    raise ValueError(f"unknown policy '{name}'; choose fcfs | srpt | dp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the simulator with a pluggable admission policy."
    )
    parser.add_argument("--duration", type=float, default=500.0)
    parser.add_argument("--arrival-rate", type=float, default=2.0)
    parser.add_argument("--max-batch-size", "-B", type=int, default=8)
    parser.add_argument("--kv-budget", "-K", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prompt-mean", type=float, default=256.0)
    parser.add_argument("--generation-mean", type=float, default=128.0)
    parser.add_argument(
        "--policy",
        choices=["fcfs", "srpt", "dp"],
        default="fcfs",
    )
    parser.add_argument(
        "--scheduling",
        choices=["fcfs", "srpt"],
        default=None,
        help="Batch-selection order. Defaults to match the policy.",
    )
    parser.add_argument("--policy-table", type=Path, default=None)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "policy_baseline.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scheduling = args.scheduling or ("srpt" if args.policy == "srpt" else "fcfs")
    policy = build_policy(
        args.policy, policy_table=args.policy_table, page_size=args.page_size
    )
    simulator = PolicySimulator(
        policy=policy,
        scheduling=scheduling,
        duration=args.duration,
        arrival_rate=args.arrival_rate,
        max_batch_size=args.max_batch_size,
        kv_budget=args.kv_budget,
        seed=args.seed,
        prompt_mean=args.prompt_mean,
        generation_mean=args.generation_mean,
    )
    records = simulator.run()
    summary = summarize(records, args.duration)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    import csv

    rows = records_as_rows(records)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"policy: {policy.name}, scheduling: {scheduling}")
    print(f"wrote {args.output}")
    for key in (
        "total_requests",
        "accepted",
        "rejected",
        "blocking_probability",
        "goodput_rps",
        "mean_ttft",
        "p99_ttft",
    ):
        print(f"{key}: {summary[key]}")


if __name__ == "__main__":
    main()
