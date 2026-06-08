"""Two-stage closed-form approximation for continuous-batching LLM inference.

This is contribution #1 of the analytical track. Distinct from
`analytical.approximation` (Adam's bulk M/M/C placeholder with Erlang-B/C)
in two ways:

  1. **State-dependent service rate.** The PS pool's drain rate is
     min(n, C) / s(min(n, C)), where s is the simulator's per-iteration
     latency curve (so the analytical and simulated systems share the same
     service mechanism). The M/M/C baseline collapses this to a single rate
     at full batch.

  2. **Two-stage decomposition.** In state n, min(n, C) requests are
     "active" in the PS pool (Stage B) and max(n - C, 0) are "waiting for a
     batch slot" in the admission queue (Stage A). The system is one
     birth-death chain on n in {0, ..., N_max}, with
     N_max = floor(K / E[kv]) the KV-implied admission cap that matches
     the simulator's KV admission test.

The stationary distribution is product-form:
    p_n = p_0 * prod_{k=1..n} lambda / mu_k,  p_0 = (sum prods)^{-1}.

Tail (p50/p95/p99) uses the exact wait CDF: an arrival that finds state n
with C <= n < N_max waits for (n + 1 - C) decode completions, each
exponential at rate mu_C. Wait is therefore Erlang-distributed conditional
on the pre-arrival state, and the TTFT CDF is the PASTA-weighted mixture.
"""
from __future__ import annotations

import math

from analytical.approximation import (
    GENERATION_SIGMA,
    MAX_UNSTABLE_WAIT_SECONDS,
    PROMPT_SIGMA,
    lognormal_expected_value,
)
from analytical.fit_service_rate import LatencyModel, get_iteration_latency
from simulator.simpy_simulator import iteration_latency


def _service_time_at(
    batch: int,
    avg_prompt: float,
    avg_generation: float,
    latency_model: LatencyModel | None = None,
) -> float:
    prefill = get_iteration_latency(
        latency_model, batch, int(round(avg_prompt)), "prefill"
    )
    decode_iter = get_iteration_latency(latency_model, batch, 1, "decode")
    return prefill + avg_generation * decode_iter


def birth_death_distribution(
    arrival_rate: float, service_rates: list[float]
) -> list[float]:
    """Stationary distribution of an M/M/1-style chain on {0, ..., N_max}.

    service_rates[i] is mu_{i+1}, the total drain rate when there are i+1
    requests in the system. Returns probabilities p[0..N_max].
    """
    products = [1.0]
    for mu_n in service_rates:
        if mu_n <= 0:
            products.append(0.0)
        else:
            products.append(products[-1] * arrival_rate / mu_n)
    total = sum(products)
    if total <= 0:
        return [1.0] + [0.0] * len(service_rates)
    return [pi / total for pi in products]


def erlang_cdf(t: float, k: int, rate: float) -> float:
    """P(Erlang(k, rate) <= t). Iterative, numerically stable for moderate k."""
    if k <= 0:
        return 1.0
    if t <= 0 or rate <= 0:
        return 0.0
    rt = rate * t
    # Survival = sum_{i=0..k-1} (rt)^i / i! * exp(-rt)
    term = math.exp(-rt)
    survival = term
    for i in range(1, k):
        term *= rt / i
        survival += term
    return max(0.0, 1.0 - survival)


def _ttft_quantile(
    percentile: float,
    *,
    p: list[float],
    active_cap: int,
    n_max: int,
    mu_full: float,
    first_token_service: float,
) -> float:
    """Quantile of TTFT (admission wait + first-token service) for an accepted
    arrival, via binary-search inversion of the wait CDF.
    """
    p_blocked = p[n_max] if n_max < len(p) else 0.0
    p_accepted = max(1.0 - p_blocked, 0.0)
    if p_accepted <= 0:
        return first_token_service + MAX_UNSTABLE_WAIT_SECONDS

    no_wait_mass = sum(p[n] for n in range(min(active_cap, n_max + 1)))
    accepted_no_wait = no_wait_mass / p_accepted
    if percentile <= accepted_no_wait:
        return first_token_service

    if mu_full <= 0:
        return first_token_service + MAX_UNSTABLE_WAIT_SECONDS

    def cdf(t: float) -> float:
        cumulative = no_wait_mass
        for n in range(active_cap, n_max):
            k = n + 1 - active_cap
            cumulative += p[n] * erlang_cdf(t, k, mu_full)
        return cumulative / p_accepted

    lo, hi = 0.0, 1.0
    while cdf(hi) < percentile and hi < MAX_UNSTABLE_WAIT_SECONDS:
        hi *= 2.0
    if cdf(hi) < percentile:
        return first_token_service + MAX_UNSTABLE_WAIT_SECONDS
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if cdf(mid) >= percentile:
            hi = mid
        else:
            lo = mid
    return first_token_service + hi


def two_stage_metrics(
    arrival_rate: float,
    max_batch_size: int,
    kv_budget: int,
    *,
    prompt_mean: float = 256.0,
    generation_mean: float = 128.0,
    latency_model: LatencyModel | None = None,
) -> dict[str, float]:
    """Two-stage closed-form approximation; see module docstring for the model."""
    avg_prompt = lognormal_expected_value(prompt_mean, PROMPT_SIGMA)
    avg_generation = lognormal_expected_value(generation_mean, GENERATION_SIGMA)
    avg_kv_per_request = avg_prompt + avg_generation

    n_max = max(1, int(kv_budget // avg_kv_per_request))
    active_cap = max(1, min(max_batch_size, n_max))

    service_rates: list[float] = []
    for n in range(1, n_max + 1):
        active = min(n, active_cap)
        s = _service_time_at(active, avg_prompt, avg_generation, latency_model)
        service_rates.append(active / s if s > 0 else 0.0)

    p = birth_death_distribution(arrival_rate, service_rates)

    blocking_probability = p[n_max] if n_max < len(p) else 1.0
    accepted_arrival_rate = arrival_rate * (1.0 - blocking_probability)
    goodput = accepted_arrival_rate

    # Mean wait in admission queue via Little's law on Stage A.
    mean_queue_length = sum(max(n - active_cap, 0) * p[n] for n in range(n_max + 1))
    if accepted_arrival_rate > 0:
        mean_admission_wait = min(
            mean_queue_length / accepted_arrival_rate, MAX_UNSTABLE_WAIT_SECONDS
        )
    else:
        mean_admission_wait = 0.0

    # First-token service at the typical active batch size when busy.
    busy_mass = 1.0 - p[0]
    if busy_mass > 1e-9:
        avg_active_when_busy = (
            sum(min(n, active_cap) * p[n] for n in range(1, n_max + 1)) / busy_mass
        )
        bar_b = max(1, int(round(avg_active_when_busy)))
    else:
        bar_b = 1
    first_token_service = (
        get_iteration_latency(latency_model, bar_b, int(round(avg_prompt)), "prefill")
        + get_iteration_latency(latency_model, bar_b, 1, "decode")
    )
    mean_ttft = mean_admission_wait + first_token_service

    # Tail via Erlang inversion. mu_full is the drain rate from a fully
    # active batch of size C; that is the rate at which slots free up while
    # the waiting queue is non-empty.
    s_full = _service_time_at(
        active_cap, avg_prompt, avg_generation, latency_model
    )
    mu_full = active_cap / s_full if s_full > 0 else 0.0

    return {
        "arrival_rate": arrival_rate,
        "max_batch_size": max_batch_size,
        "kv_budget": kv_budget,
        "mean_ttft": mean_ttft,
        "p50_ttft": _ttft_quantile(
            0.50,
            p=p,
            active_cap=active_cap,
            n_max=n_max,
            mu_full=mu_full,
            first_token_service=first_token_service,
        ),
        "p95_ttft": _ttft_quantile(
            0.95,
            p=p,
            active_cap=active_cap,
            n_max=n_max,
            mu_full=mu_full,
            first_token_service=first_token_service,
        ),
        "p99_ttft": _ttft_quantile(
            0.99,
            p=p,
            active_cap=active_cap,
            n_max=n_max,
            mu_full=mu_full,
            first_token_service=first_token_service,
        ),
        "goodput": goodput,
        "blocking_probability": blocking_probability,
    }
