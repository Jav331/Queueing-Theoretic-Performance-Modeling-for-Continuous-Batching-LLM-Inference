from __future__ import annotations

import math

from simulator.simpy_simulator import iteration_latency


PROMPT_SIGMA = 0.55
GENERATION_SIGMA = 0.70
MAX_UNSTABLE_WAIT_SECONDS = 60.0


def lognormal_expected_value(median: float, sigma: float) -> float:
    """The simulator samples lognormal(log(median), sigma)."""
    return math.exp(math.log(median) + 0.5 * sigma**2)


def erlang_b(offered_load: float, capacity: int) -> float:
    """Blocking probability for a simple finite-capacity loss system."""
    if capacity <= 0:
        return 1.0

    blocking = 1.0
    for n in range(1, capacity + 1):
        blocking = (offered_load * blocking) / (n + offered_load * blocking)
    return blocking


def estimate_effective_capacity(
    max_batch_size: int,
    kv_budget: int,
    avg_kv_per_request: float,
) -> int:
    """Convert batch and KV limits into a simple effective concurrency cap."""
    kv_limited_concurrency = int(kv_budget // avg_kv_per_request)
    return max(1, min(max_batch_size, kv_limited_concurrency))


def erlang_c_wait(arrival_rate: float, service_time: float, servers: int) -> float:
    """Mean queueing delay for an M/M/C queue using the Erlang-C formula."""
    if servers <= 0:
        return math.inf
    if arrival_rate <= 0 or service_time <= 0:
        return 0.0

    service_rate = 1.0 / service_time
    offered_load = arrival_rate / service_rate
    utilization = offered_load / servers
    if utilization >= 1.0:
        return MAX_UNSTABLE_WAIT_SECONDS

    partial_sum = sum(offered_load**n / math.factorial(n) for n in range(servers))
    tail = (offered_load**servers / math.factorial(servers)) / (1.0 - utilization)
    p0 = 1.0 / (partial_sum + tail)
    p_wait = tail * p0
    return p_wait / (servers * service_rate - arrival_rate)


def percentile_from_exponential_tail(
    prefill_service_time: float,
    queueing_delay: float,
    utilization: float,
    percentile: float,
) -> float:
    """Approximate TTFT percentiles with a utilization-sensitive tail."""
    rho = min(max(utilization, 0.0), 0.999)
    tail_scale = queueing_delay * (1.0 + 2.0 * rho)
    return prefill_service_time - math.log(1.0 - percentile) * tail_scale


def approximate_metrics(
    arrival_rate: float,
    max_batch_size: int,
    kv_budget: int,
    *,
    prompt_mean: float = 256.0,
    generation_mean: float = 128.0,
) -> dict[str, float]:
    """Return a closed-form approximation for the baseline simulator.

    Assumptions:
    - Average request sizes stand in for the full prompt/generation distributions.
    - KV capacity becomes an effective concurrency cap.
    - Continuous batching is approximated as C parallel service positions.
    - Queueing delay follows an M/M/C approximation.
    - Blocking follows an Erlang-B finite-capacity loss approximation.
    - TTFT has a deterministic prefill/first-decode component plus a
      utilization-sensitive exponential queueing tail.

    This is still only a first closed-form baseline, not the final Markov-chain
    model over queue length, active batch size, and KV occupancy.
    """
    avg_prompt = lognormal_expected_value(prompt_mean, PROMPT_SIGMA)
    avg_generation = lognormal_expected_value(generation_mean, GENERATION_SIGMA)
    avg_kv_per_request = avg_prompt + avg_generation

    concurrency = estimate_effective_capacity(
        max_batch_size, kv_budget, avg_kv_per_request
    )

    # The simulator's latency curve is state-dependent. We approximate the
    # typical active batch as increasing with offered load instead of always
    # assuming the batch is full.
    single_prefill = iteration_latency(1, int(round(avg_prompt)), "prefill")
    single_decode = iteration_latency(1, 1, "decode")
    single_service_time = single_prefill + avg_generation * single_decode
    initial_rho = min(arrival_rate * single_service_time / concurrency, 1.0)
    effective_batch = max(
        1, min(concurrency, round(1 + (concurrency - 1) * initial_rho))
    )

    # TTFT separates queueing delay from the first-token service component.
    # Prefill uses average prompt tokens for the estimated active batch; the
    # first decode iteration produces the first generated token.
    prefill_time = iteration_latency(concurrency, int(round(avg_prompt)), "prefill")
    batch_prefill_time = iteration_latency(
        effective_batch, int(round(avg_prompt * effective_batch)), "prefill"
    )
    decode_step_time = iteration_latency(effective_batch, 1, "decode")
    first_token_service = prefill_time + decode_step_time
    service_time = batch_prefill_time + avg_generation * decode_step_time

    service_capacity = concurrency / service_time if service_time > 0 else 0.0
    offered_load = arrival_rate * service_time
    offered_utilization = offered_load / concurrency if concurrency > 0 else 1.0

    # Erlang-B treats the effective concurrency C as a finite loss system. This
    # is a coarse stand-in for KV admission limits, not a KV-state Markov chain.
    blocking_probability = erlang_b(offered_load, concurrency)
    if arrival_rate > 0 and service_capacity < arrival_rate:
        overload_blocking = 1.0 - service_capacity / arrival_rate
        blocking_probability = max(blocking_probability, overload_blocking)
    blocking_probability = min(max(blocking_probability, 0.0), 1.0)

    accepted_arrival_rate = arrival_rate * (1.0 - blocking_probability)
    goodput = min(accepted_arrival_rate, service_capacity)

    # Blocking removes excess traffic before the queueing approximation. Use
    # post-admission load for TTFT tails so overloaded cases stay finite instead
    # of double-counting overload as both blocking and very large waiting.
    stable_arrival_rate = min(accepted_arrival_rate, service_capacity * 0.95)
    admitted_utilization = (
        stable_arrival_rate * service_time / concurrency if concurrency > 0 else 1.0
    )
    raw_wait = erlang_c_wait(stable_arrival_rate, service_time, concurrency)
    if offered_utilization >= 1.0 and blocking_probability <= 0.0:
        raw_wait = MAX_UNSTABLE_WAIT_SECONDS
    wait = min(raw_wait, MAX_UNSTABLE_WAIT_SECONDS)

    mean_ttft = first_token_service + wait
    p50_ttft = percentile_from_exponential_tail(
        first_token_service, wait, admitted_utilization, 0.50
    )
    p95_ttft = percentile_from_exponential_tail(
        first_token_service, wait, admitted_utilization, 0.95
    )
    p99_ttft = percentile_from_exponential_tail(
        first_token_service, wait, admitted_utilization, 0.99
    )

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
    }
