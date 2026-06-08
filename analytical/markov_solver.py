"""Multi-class continuous-time Markov chain solver (Phase 2).

State: n = (n_1, ..., n_K) where n_c is the number of class-c requests in
system (active or waiting). The chain is over the feasible set
    {n : sum_c n_c * d_c <= K_pages}
with d_c the per-class KV demand in pages (see analytical.state_space).

Transitions from state n:

  Arrival of class c, rate lambda * pi_c:
    n -> n + e_c  if n + e_c is KV-feasible; else blocked (lost arrival).

  Completion of class c, rate mu_c(n):
    n -> n - e_c
    where mu_c(n) = (active / s(active)) * (n_c / total)
    and  active = min(sum n, B),  s(b) = iteration_latency-derived service
    time per request at batch b. Per-class share is the continuous
    proportional-sharing approximation (PS over heterogeneous classes).

The stationary distribution is solved numerically from pi Q = 0, sum pi = 1
via a small sparse linear system. Tail TTFT uses the PASTA + Erlang-CDF
machinery from closed_form, evaluated per (arrival state, arrival class).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from analytical.approximation import MAX_UNSTABLE_WAIT_SECONDS
from analytical.closed_form import erlang_cdf
from analytical.fit_service_rate import LatencyModel, get_iteration_latency
from analytical.state_space import (
    DEFAULT_PAGE_SIZE,
    RequestClass,
    build_request_classes,
    enumerate_states,
    state_index_map,
    total_in_system,
)
from simulator.simpy_simulator import iteration_latency


def _service_time(
    batch: int,
    classes: list[RequestClass],
    latency_model: LatencyModel | None = None,
) -> float:
    if batch <= 0:
        return float("inf")
    # Use the unconditional E[prompt], E[generation] averaged over the
    # arrival mix so the service rate matches the simulator's behavior.
    avg_prompt = sum(c.expected_prompt * c.arrival_share for c in classes)
    avg_generation = sum(c.expected_generation * c.arrival_share for c in classes)
    prefill = get_iteration_latency(
        latency_model, batch, int(round(avg_prompt)), "prefill"
    )
    decode_iter = get_iteration_latency(latency_model, batch, 1, "decode")
    return prefill + avg_generation * decode_iter


def build_generator(
    arrival_rate: float,
    max_batch_size: int,
    kv_budget_pages: int,
    classes: list[RequestClass],
    latency_model: LatencyModel | None = None,
) -> tuple[sp.csr_matrix, list[tuple[int, ...]], dict[tuple[int, ...], int]]:
    """Construct the sparse infinitesimal generator Q for the multi-class chain.

    Returns Q (shape (S, S)), the state list, and the index map.
    """
    states = enumerate_states(classes, kv_budget_pages)
    index = state_index_map(states)
    n_classes = len(classes)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    arrival_rates = [arrival_rate * c.arrival_share for c in classes]
    demands = [c.kv_pages for c in classes]

    for i, state in enumerate(states):
        kv_now = sum(n * d for n, d in zip(state, demands))
        diag = 0.0

        # Arrivals.
        for c in range(n_classes):
            kv_after = kv_now + demands[c]
            if kv_after <= kv_budget_pages:
                successor = tuple(
                    state[k] + (1 if k == c else 0) for k in range(n_classes)
                )
                j = index[successor]
                rows.append(i)
                cols.append(j)
                data.append(arrival_rates[c])
                diag += arrival_rates[c]
            # else: arrival blocked, contributes nothing to Q.

        # Completions.
        total = total_in_system(state)
        if total > 0:
            active = min(total, max_batch_size)
            s = _service_time(active, classes, latency_model)
            total_rate = active / s if s > 0 else 0.0
            for c in range(n_classes):
                if state[c] == 0:
                    continue
                rate_c = total_rate * (state[c] / total)
                successor = tuple(
                    state[k] - (1 if k == c else 0) for k in range(n_classes)
                )
                j = index[successor]
                rows.append(i)
                cols.append(j)
                data.append(rate_c)
                diag += rate_c

        rows.append(i)
        cols.append(i)
        data.append(-diag)

    n_states = len(states)
    q_matrix = sp.csr_matrix(
        (data, (rows, cols)), shape=(n_states, n_states)
    )
    return q_matrix, states, index


def solve_stationary(q_matrix: sp.csr_matrix) -> np.ndarray:
    """Solve pi Q = 0, sum pi = 1 for the stationary distribution."""
    n_states = q_matrix.shape[0]
    # We want left null vector of Q, equivalently right null of Q^T.
    a = q_matrix.T.tolil()
    a[-1, :] = 1.0
    a = a.tocsr()
    b = np.zeros(n_states)
    b[-1] = 1.0
    pi = spla.spsolve(a, b)
    # Numerical hygiene: clip tiny negatives and renormalize.
    pi = np.clip(pi, 0.0, None)
    s = pi.sum()
    if s > 0:
        pi /= s
    return pi


def _ttft_cdf(
    t: float,
    *,
    pi: np.ndarray,
    states: list[tuple[int, ...]],
    demands: list[int],
    kv_budget_pages: int,
    max_batch_size: int,
    mu_full: float,
    first_token_service: float,
    arrival_rates: list[float],
    accepted_rate_total: float,
) -> float:
    """CDF of TTFT for an accepted arrival, averaged over (state, class)."""
    if accepted_rate_total <= 0:
        return 0.0
    if t < first_token_service:
        return 0.0
    wait_t = t - first_token_service
    cumulative = 0.0
    for i, state in enumerate(states):
        kv_now = sum(n * d for n, d in zip(state, demands))
        total = sum(state)
        for c, lam_c in enumerate(arrival_rates):
            if kv_now + demands[c] > kv_budget_pages:
                continue  # blocked
            weight = pi[i] * lam_c / accepted_rate_total
            if total < max_batch_size:
                cumulative += weight  # no wait
            elif mu_full > 0:
                k = total - max_batch_size + 1
                cumulative += weight * erlang_cdf(wait_t, k, mu_full)
            # else mu_full == 0 contributes nothing (infinite wait).
    return cumulative


def _quantile_via_inversion(
    percentile: float,
    cdf_fn,
    floor: float,
    cap: float = MAX_UNSTABLE_WAIT_SECONDS,
) -> float:
    if cdf_fn(floor) >= percentile:
        return floor
    upper_cap = floor + cap
    lo, hi = floor, min(upper_cap, max(floor + 1.0, floor * 2.0))
    while cdf_fn(hi) < percentile and hi < upper_cap:
        hi = min(upper_cap, hi * 2.0)
    if cdf_fn(hi) < percentile:
        return upper_cap
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if cdf_fn(mid) >= percentile:
            hi = mid
        else:
            lo = mid
    return hi


def markov_metrics(
    arrival_rate: float,
    max_batch_size: int,
    kv_budget: int,
    *,
    prompt_mean: float = 256.0,
    generation_mean: float = 128.0,
    n_classes: int = 2,
    page_size: int = DEFAULT_PAGE_SIZE,
    classes: Optional[list[RequestClass]] = None,
    latency_model: LatencyModel | None = None,
) -> dict[str, float]:
    """Solve the multi-class chain at (lambda, B, K) and extract metrics."""
    if classes is None:
        classes = build_request_classes(
            n_classes=n_classes,
            prompt_mean=prompt_mean,
            generation_mean=generation_mean,
            page_size=page_size,
        )
    kv_budget_pages = max(1, kv_budget // page_size)
    demands = [c.kv_pages for c in classes]
    arrival_rates = [arrival_rate * c.arrival_share for c in classes]

    q_matrix, states, _ = build_generator(
        arrival_rate, max_batch_size, kv_budget_pages, classes, latency_model
    )
    pi = solve_stationary(q_matrix)

    # Blocking: weighted over classes by arrival share.
    blocking_probability = 0.0
    for i, state in enumerate(states):
        kv_now = sum(n * d for n, d in zip(state, demands))
        for c, share in enumerate([cls.arrival_share for cls in classes]):
            if kv_now + demands[c] > kv_budget_pages:
                blocking_probability += pi[i] * share

    accepted_rate_total = arrival_rate * (1.0 - blocking_probability)
    goodput = accepted_rate_total

    # First-token service at typical active batch given the system is busy.
    busy_mass = 1.0 - pi[0]
    if busy_mass > 1e-12:
        avg_active = 0.0
        for i, state in enumerate(states):
            total = sum(state)
            if total == 0:
                continue
            avg_active += pi[i] * min(total, max_batch_size)
        avg_active /= busy_mass
        bar_b = max(1, int(round(avg_active)))
    else:
        bar_b = 1
    avg_prompt = sum(c.expected_prompt * c.arrival_share for c in classes)
    first_token_service = (
        get_iteration_latency(latency_model, bar_b, int(round(avg_prompt)), "prefill")
        + get_iteration_latency(latency_model, bar_b, 1, "decode")
    )

    # Mean wait via Little's law on the (post-arrival) queue length.
    mean_queue = 0.0
    for i, state in enumerate(states):
        total = sum(state)
        if total > max_batch_size:
            mean_queue += pi[i] * (total - max_batch_size)
    mean_wait = (
        min(mean_queue / accepted_rate_total, MAX_UNSTABLE_WAIT_SECONDS)
        if accepted_rate_total > 0
        else 0.0
    )
    mean_ttft = mean_wait + first_token_service

    # Tail TTFT via PASTA-weighted Erlang-CDF inversion.
    s_full = _service_time(max_batch_size, classes, latency_model)
    mu_full = max_batch_size / s_full if s_full > 0 else 0.0

    def cdf(t: float) -> float:
        return _ttft_cdf(
            t,
            pi=pi,
            states=states,
            demands=demands,
            kv_budget_pages=kv_budget_pages,
            max_batch_size=max_batch_size,
            mu_full=mu_full,
            first_token_service=first_token_service,
            arrival_rates=arrival_rates,
            accepted_rate_total=accepted_rate_total,
        )

    p50_ttft = _quantile_via_inversion(0.50, cdf, first_token_service)
    p95_ttft = _quantile_via_inversion(0.95, cdf, first_token_service)
    p99_ttft = _quantile_via_inversion(0.99, cdf, first_token_service)

    return {
        "arrival_rate": arrival_rate,
        "max_batch_size": max_batch_size,
        "kv_budget": kv_budget,
        "mean_ttft": mean_ttft,
        "p50_ttft": p50_ttft,
        "p95_ttft": p95_ttft,
        "p99_ttft": p99_ttft,
        "goodput": goodput,
        "blocking_probability": blocking_probability,
        "n_states": len(states),
    }
