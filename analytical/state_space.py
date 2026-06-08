"""State-space utilities for the multi-class Markov chain (Phase 2).

The 1D birth-death chain in `analytical.closed_form` collapses every request
to E[kv]. The proposal calls for "a Markov chain over (queue length, active
batch size, aggregate KV occupancy)" which only adds information beyond the
1D model if requests are heterogeneous in their KV demand. We split the
lognormal prompt+generation distribution into K size classes (default 2:
short, long) so the chain state (n_1, ..., n_K) captures both queue/batch
occupancy AND aggregate KV occupancy via kv = sum_c n_c * d_c.

KV demands d_c are quantized to page granularity (default 16 tokens per
page, matching vLLM's PagedAttention default). The KV constraint becomes
sum_c n_c * d_c_pages <= K / page_size, which the state enumeration uses
to prune infeasible states.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from analytical.approximation import (
    GENERATION_SIGMA,
    PROMPT_SIGMA,
    lognormal_expected_value,
)


DEFAULT_PAGE_SIZE = 16  # vLLM PagedAttention default (tokens per page)


@dataclass(frozen=True)
class RequestClass:
    """A discrete class in the multi-class request mix."""

    name: str
    kv_pages: int  # KV demand in pages
    expected_generation: float  # E[generation_len] for this class
    expected_prompt: float  # E[prompt_len] for this class
    arrival_share: float  # probability of an arrival belonging to this class


def lognormal_quantile(median: float, sigma: float, q: float) -> float:
    """Quantile q (in [0, 1]) of Lognormal(log median, sigma)."""
    # Inverse of standard normal CDF at q via the Beasley-Springer-Moro
    # approximation. SciPy's norm.ppf would do this but we avoid the import
    # to keep this module dependency-light.
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    # Acklam's inverse normal CDF coefficients.
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if q < p_low:
        r = math.sqrt(-2.0 * math.log(q))
        z = (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / (
            (((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0
        )
    elif q <= p_high:
        r = q - 0.5
        r2 = r * r
        z = ((((a[0] * r2 + a[1]) * r2 + a[2]) * r2 + a[3]) * r2 + a[4]) * r2 + a[5]
        z = z * r / (((((b[0] * r2 + b[1]) * r2 + b[2]) * r2 + b[3]) * r2 + b[4]) * r2 + 1.0)
    else:
        r = math.sqrt(-2.0 * math.log(1.0 - q))
        z = -(((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / (
            (((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0
        )
    return math.exp(math.log(median) + sigma * z)


def lognormal_conditional_mean(median: float, sigma: float, q_lo: float, q_hi: float) -> float:
    """E[X | q_lo < F(X) <= q_hi] for X ~ Lognormal(log median, sigma).

    Exact closed-form via the standard normal CDF; uses the same Acklam
    inverse for boundary quantiles and the Mills-ratio identity
    E[X | a < Z <= b] = exp(mu + sigma^2 / 2) * (Phi(b - sigma) - Phi(a - sigma)) / (Phi(b) - Phi(a))
    """
    if q_hi <= q_lo:
        return lognormal_quantile(median, sigma, (q_hi + q_lo) / 2.0)

    def norm_cdf(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    # Boundary z-values from the inverse normal at q_lo / q_hi.
    if q_lo <= 0.0:
        z_lo = -10.0
    else:
        z_lo = (math.log(lognormal_quantile(median, sigma, q_lo)) - math.log(median)) / sigma
    if q_hi >= 1.0:
        z_hi = 10.0
    else:
        z_hi = (math.log(lognormal_quantile(median, sigma, q_hi)) - math.log(median)) / sigma

    cdf_diff = norm_cdf(z_hi) - norm_cdf(z_lo)
    if cdf_diff <= 0.0:
        return lognormal_quantile(median, sigma, (q_hi + q_lo) / 2.0)
    return (
        math.exp(math.log(median) + 0.5 * sigma**2)
        * (norm_cdf(z_hi - sigma) - norm_cdf(z_lo - sigma))
        / cdf_diff
    )


def build_request_classes(
    *,
    n_classes: int = 2,
    prompt_mean: float = 256.0,
    generation_mean: float = 128.0,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> list[RequestClass]:
    """Split the lognormal prompt+generation distribution into n_classes by
    equal-probability bins of the joint kv = prompt + generation."""
    if n_classes < 1:
        raise ValueError("n_classes must be >= 1")
    classes: list[RequestClass] = []
    edges = [i / n_classes for i in range(n_classes + 1)]
    for idx in range(n_classes):
        q_lo, q_hi = edges[idx], edges[idx + 1]
        avg_prompt = lognormal_conditional_mean(prompt_mean, PROMPT_SIGMA, q_lo, q_hi)
        avg_generation = lognormal_conditional_mean(
            generation_mean, GENERATION_SIGMA, q_lo, q_hi
        )
        kv_tokens = avg_prompt + avg_generation
        kv_pages = max(1, int(math.ceil(kv_tokens / page_size)))
        classes.append(
            RequestClass(
                name=f"c{idx}",
                kv_pages=kv_pages,
                expected_generation=avg_generation,
                expected_prompt=avg_prompt,
                arrival_share=1.0 / n_classes,
            )
        )
    return classes


def enumerate_states(
    classes: list[RequestClass], kv_budget_pages: int
) -> list[tuple[int, ...]]:
    """All non-negative integer tuples (n_1, ..., n_K) with
    sum_c n_c * d_c <= kv_budget_pages."""
    n_classes = len(classes)
    demands = [c.kv_pages for c in classes]

    states: list[tuple[int, ...]] = []

    def recurse(idx: int, partial: list[int], remaining: int) -> None:
        if idx == n_classes:
            states.append(tuple(partial))
            return
        max_n = remaining // demands[idx]
        for n in range(max_n + 1):
            partial.append(n)
            recurse(idx + 1, partial, remaining - n * demands[idx])
            partial.pop()

    recurse(0, [], kv_budget_pages)
    return states


def state_index_map(states: list[tuple[int, ...]]) -> dict[tuple[int, ...], int]:
    return {state: i for i, state in enumerate(states)}


def total_in_system(state: tuple[int, ...]) -> int:
    return sum(state)


def kv_pages_occupied(state: tuple[int, ...], classes: list[RequestClass]) -> int:
    return sum(n * c.kv_pages for n, c in zip(state, classes))
