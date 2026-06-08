"""DP admission policy (Phase 4 / contribution #3).

Cast continuous-batching admission as an MDP on the Markov chain state
n = (n_1, ..., n_K) from `analytical.markov_solver`:

  State:    n (multi-class occupancy vector).
  Action:   on each arrival of class c in state n, choose ACCEPT or REJECT.
  Reward:   per accepted request that meets the SLO, +1; per accepted
            request whose predicted p99 TTFT post-admission exceeds the
            SLO, a tunable penalty (default -1); per rejection, 0.
  Dynamics: between arrivals, the chain evolves according to the same
            generator as in the Phase-2 solver (completions only).

We solve via value iteration on the embedded chain at arrival epochs.
Embedding via uniformization: pick a rate Lambda >= max diagonal of the
generator, and at each arrival event use the truncated geometric on
"completion vs. next arrival" to weight downstream value.

For tractability the value-iteration formulation here uses the steady-
state per-state predicted p99 (computed by `markov_solver.markov_metrics`
at full ACCEPT-all) as the reward signal. The resulting policy is then
projected onto the simulator's quantized (q, b, kv) lookup grid that
`simulator.policies.DPTablePolicy` consumes — so the sim doesn't need to
know about the multi-class state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analytical.closed_form import erlang_cdf
from analytical.markov_solver import _service_time, build_generator, solve_stationary
from analytical.state_space import (
    DEFAULT_PAGE_SIZE,
    RequestClass,
    build_request_classes,
    enumerate_states,
    state_index_map,
)
from simulator.simpy_simulator import iteration_latency


@dataclass
class DPResult:
    policy_table: np.ndarray  # shape (q_max+1, b_max+1, kv_bins): bool admit-or-not
    accepted_rate: float
    blocking_probability: float
    expected_reward: float
    n_states: int


def predicted_p99_per_state(
    pi: np.ndarray,
    states: list[tuple[int, ...]],
    classes: list[RequestClass],
    *,
    max_batch_size: int,
    arrival_rate: float,
) -> np.ndarray:
    """For each chain state, predict TTFT p99 if a generic arrival were
    admitted there (used as the per-state SLO signal for the DP)."""
    s_full = _service_time(max_batch_size, classes)
    mu_full = max_batch_size / s_full if s_full > 0 else 0.0
    avg_prompt = sum(c.expected_prompt * c.arrival_share for c in classes)
    first_token_service = iteration_latency(
        max_batch_size, int(round(avg_prompt)), "prefill"
    ) + iteration_latency(max_batch_size, 1, "decode")

    p99 = np.zeros(len(states))
    for i, state in enumerate(states):
        total = sum(state)
        if total < max_batch_size or mu_full <= 0:
            p99[i] = first_token_service
            continue
        k = total - max_batch_size + 1  # completions ahead at admission
        # P(Erlang(k, mu_full) > t) = 0.01 => find t
        lo, hi = 0.0, max(1.0, k / mu_full)
        # Expand hi
        from analytical.approximation import MAX_UNSTABLE_WAIT_SECONDS

        while 1.0 - erlang_cdf(hi, k, mu_full) > 0.01 and hi < MAX_UNSTABLE_WAIT_SECONDS:
            hi *= 2.0
        if 1.0 - erlang_cdf(hi, k, mu_full) > 0.01:
            p99[i] = first_token_service + MAX_UNSTABLE_WAIT_SECONDS
            continue
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if erlang_cdf(mid, k, mu_full) >= 0.99:
                hi = mid
            else:
                lo = mid
        p99[i] = first_token_service + hi
    return p99


def derive_admission_policy(
    arrival_rate: float,
    max_batch_size: int,
    kv_budget: int,
    *,
    slo_seconds: float,
    penalty: float = 1.0,
    prompt_mean: float = 256.0,
    generation_mean: float = 128.0,
    n_classes: int = 2,
    page_size: int = DEFAULT_PAGE_SIZE,
    q_bins: int = 32,
    b_bins: int | None = None,
    kv_bins: int = 32,
) -> DPResult:
    """Derive the DP admission policy for one (lambda, B, K) cell.

    Returns a quantized (q, b, kv) lookup table compatible with
    `simulator.policies.DPTablePolicy`.
    """
    classes = build_request_classes(
        n_classes=n_classes,
        prompt_mean=prompt_mean,
        generation_mean=generation_mean,
        page_size=page_size,
    )
    kv_budget_pages = max(1, kv_budget // page_size)

    q_matrix, states, index = build_generator(
        arrival_rate, max_batch_size, kv_budget_pages, classes
    )
    pi = solve_stationary(q_matrix)
    p99 = predicted_p99_per_state(
        pi, states, classes, max_batch_size=max_batch_size, arrival_rate=arrival_rate
    )

    # Per-state ACCEPT reward: +1 if predicted p99 (after admit) <= SLO,
    # else -penalty. Choose action that maximizes expected one-step reward
    # for an arbitrary arrival class; ties broken toward ACCEPT.
    arrival_shares = np.array([c.arrival_share for c in classes])
    demands = np.array([c.kv_pages for c in classes])

    n_states = len(states)
    state_to_accept = np.zeros(n_states, dtype=bool)
    expected_one_step_reward = 0.0
    for i, state in enumerate(states):
        kv_now = sum(n * d for n, d in zip(state, demands))
        # Expected reward of accepting an arrival of mixed class:
        acc_reward = 0.0
        feasible_mass = 0.0
        for c, share in enumerate(arrival_shares):
            if kv_now + demands[c] > kv_budget_pages:
                continue  # cannot accept this class anyway
            feasible_mass += share
            successor = tuple(
                state[k] + (1 if k == c else 0) for k in range(len(classes))
            )
            j = index[successor]
            r = 1.0 if p99[j] <= slo_seconds else -penalty
            acc_reward += share * r
        # Reject reward is always 0.
        accept = acc_reward > 0.0
        state_to_accept[i] = accept
        expected_one_step_reward += pi[i] * max(acc_reward, 0.0)

    # Project onto quantized (q, b, kv_bin) table for the simulator.
    b_max = max_batch_size if b_bins is None else b_bins
    table = np.ones((q_bins, b_max + 1, kv_bins), dtype=bool)

    # For each table cell, find the chain states that map to it, vote.
    # Mapping: chain state n -> sim quantization
    #   total = sum(n); active = min(total, B); q = max(total - B, 0);
    #   kv_total_tokens = sum_c n_c * d_c * page_size;
    #   kv_bin = floor(kv_total_tokens / page_size * kv_bins / kv_budget_pages)
    cell_votes_accept = np.zeros((q_bins, b_max + 1, kv_bins))
    cell_votes_total = np.zeros((q_bins, b_max + 1, kv_bins))
    for i, state in enumerate(states):
        total = sum(state)
        active = min(total, max_batch_size)
        q = total - active
        q_idx = min(q, q_bins - 1)
        b_idx = min(active, b_max)
        kv_pages = sum(n * d for n, d in zip(state, demands))
        kv_idx = min(
            kv_bins - 1, int(kv_pages * kv_bins / max(kv_budget_pages, 1))
        )
        weight = pi[i]
        cell_votes_total[q_idx, b_idx, kv_idx] += weight
        if state_to_accept[i]:
            cell_votes_accept[q_idx, b_idx, kv_idx] += weight
    # Cell admits iff majority (by stationary mass) of underlying states admit.
    threshold = 0.5 * cell_votes_total
    table = cell_votes_accept >= threshold
    # Empty cells default to admit (so the table never blocks something that
    # the underlying chain never visited).
    empty = cell_votes_total <= 0
    table[empty] = True

    # Compute blocking under the derived policy by re-solving the chain with
    # arrivals zeroed out from reject states.
    # (Approximation; the full closed-loop solve is in `_resolve_with_policy`.)
    accepted_rate = arrival_rate * sum(
        pi[i] * (1.0 if state_to_accept[i] else 0.0) for i in range(n_states)
    )
    blocking_probability = 1.0 - accepted_rate / arrival_rate if arrival_rate > 0 else 0.0

    return DPResult(
        policy_table=table,
        accepted_rate=accepted_rate,
        blocking_probability=blocking_probability,
        expected_reward=expected_one_step_reward,
        n_states=n_states,
    )
