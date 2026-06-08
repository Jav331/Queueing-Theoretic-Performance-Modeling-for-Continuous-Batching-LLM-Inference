"""Admission policies pluggable into the policy_simulator (Phase 3).

Adam's `simulator/simpy_simulator.py` bakes a single FCFS-with-KV-check
admission rule into `_arrival_process`. Phase 3 of the analytical track
adds a thin policy abstraction WITHOUT modifying that file: this module
defines the policy interface + three concrete policies, and
`simulator.policy_simulator.PolicySimulator` subclasses Adam's simulator to
consult an injected policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from simulator.simpy_simulator import RequestRecord


@dataclass
class SimState:
    """Snapshot of the simulator state passed to admission policies."""

    reserved_kv: int
    kv_budget: int
    active_count: int
    waiting_count: int
    max_batch_size: int


class AdmissionPolicy(Protocol):
    name: str

    def admit(self, candidate: RequestRecord, state: SimState) -> bool: ...


class FCFSPolicy:
    """Same admission rule as Adam's baseline simulator: admit iff KV fits."""

    name = "fcfs"

    def admit(self, candidate: RequestRecord, state: SimState) -> bool:
        return state.reserved_kv + candidate.kv_required <= state.kv_budget


class SRPTPolicy:
    """KV-feasible admission; pairs with SRPT batch fill in PolicySimulator.

    Admission is identical to FCFS — SRPT here is about *batch selection*
    (shortest waiting request goes into active batch first), not arrival
    admission, because all admitted requests have already reserved KV.
    """

    name = "srpt"

    def admit(self, candidate: RequestRecord, state: SimState) -> bool:
        return state.reserved_kv + candidate.kv_required <= state.kv_budget


class DPTablePolicy:
    """Accept/reject lookup table indexed by quantized (queue, batch, kv) state.

    The table is produced by `analytical.dp_policy` (Phase 4). Indices come
    from clipping the live sim state into the table's shape; arrivals are
    rejected on the cheap side if the table says so OR if the KV admission
    test fails.
    """

    name = "dp"

    def __init__(self, table_path: Path, page_size: int = 16):
        self.table = np.load(table_path)
        self.page_size = page_size
        if self.table.ndim != 3:
            raise ValueError(
                f"DP policy table must be 3-D (q, b, kv_bin); got {self.table.shape}"
            )

    def admit(self, candidate: RequestRecord, state: SimState) -> bool:
        if state.reserved_kv + candidate.kv_required > state.kv_budget:
            return False
        q_size, b_size, kv_size = self.table.shape
        q_idx = min(state.waiting_count, q_size - 1)
        b_idx = min(state.active_count, b_size - 1)
        kv_budget_pages = max(1, state.kv_budget // self.page_size)
        kv_pages = state.reserved_kv // self.page_size
        kv_idx = min(kv_size - 1, int(kv_pages * kv_size / kv_budget_pages))
        return bool(self.table[q_idx, b_idx, kv_idx])
