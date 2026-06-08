# Report Results Summary

This file is a report-facing summary of the current SimPy, analytical, and
Modal/vLLM results. It is written as source material for the final project
report rather than as a raw experiment log.

## Experiment Artifacts

| artifact | role |
|---|---|
| `results/sweep_summary.csv` | SimPy continuous-batching sweep, averaged over seeds |
| `results/comparison_summary.csv` | analytical approximation vs SimPy relative-error table |
| `results/vllm_sweep_summary.csv` | Modal/vLLM serving sweep on Qwen2.5-1.5B-Instruct |
| `results/vllm_sweep_metadata.json` | vLLM version, flags, realized KV cache, TPOT, throughput, preemption metadata |
| `results/latency_curve.json` | Modal microbenchmark fit for decode step latency |
| `data/sharegpt_trace.parquet` | canonical ShareGPT replay trace |

## SimPy and Analytical Comparison

The analytical baseline is compared against the SimPy simulator on the same
grid in `results/comparison_summary.csv`. The simulator grid used here has 48
rows: arrival rates 0.5, 1, 2, 3 req/s; max batch sizes 1, 4, 8; and KV budgets
2048, 4096, 8192, 16384.

| metric | mean relative error | median relative error | interpretation |
|---|---:|---:|---|
| mean TTFT | 0.712 | 0.325 | coarse model captures order of magnitude, but queueing transients are imperfect |
| p95 TTFT | 1.821 | 0.295 | mean error is dominated by overload/tail regimes |
| p99 TTFT | 1.894 | 0.453 | simple exponential tail is useful but not a final tail model |
| goodput | 0.177 | 0.106 | throughput is modeled much better than latency tails |
| blocking probability | 4.410 | 0.056 | relative error inflates when true blocking is near zero |

Report takeaway: the first closed-form approximation is a useful baseline, but
it should be presented as a sanity-check model rather than the final Markov
solver. Its strongest result is goodput prediction; its weakest result is
tail-latency accuracy under saturation.

## vLLM Hardware Sweep

The vLLM sweep contains 64 measured rows:

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- vLLM version: `0.22.0`
- GPU class: Modal A10G
- Arrival rates: 0.5, 1, 2, 3, 4, 5, 6, 8 req/s
- Max sequence counts: 64, 128
- GPU memory utilization settings: 0.3, 0.5, 0.7, 0.9
- Realized KV cache range: 104,336 to 614,736 tokens
- Warmup requests per benchmark: 8
- Measured requests per row: 300

### vLLM Aggregate Results by Arrival Rate

| arrival rate (req/s) | mean TTFT (s) | max p99 TTFT (s) | mean goodput (req/s) | mean TPOT (ms/token) |
|---:|---:|---:|---:|---:|
| 0.5 | 0.0365 | 0.1455 | 0.4945 | 8.29 |
| 1.0 | 0.0335 | 0.1121 | 0.9780 | 8.40 |
| 2.0 | 0.0367 | 0.1571 | 1.9132 | 8.63 |
| 3.0 | 0.0380 | 0.1554 | 2.8074 | 8.80 |
| 4.0 | 0.0380 | 0.1418 | 3.6639 | 9.00 |
| 5.0 | 0.0393 | 0.1137 | 4.4829 | 9.22 |
| 6.0 | 0.0417 | 0.1396 | 5.2667 | 9.52 |
| 8.0 | 0.0461 | 0.1850 | 6.7319 | 10.21 |

### vLLM Observations

- Goodput rises approximately linearly with offered load over the measured
  range, reaching 6.74 req/s at 8 req/s offered load.
- TTFT remains low on this small model: mean TTFT stays below 61 ms and p99
  TTFT stays below 185 ms across all measured rows.
- TPOT increases with load from about 8.3 ms/token to about 10.2 ms/token,
  showing the expected decode-step contention as active concurrency increases.
- Realized KV changed substantially with memory-utilization settings, but this
  measured grid still did not trigger request failures or vLLM preemptions:
  blocking probability and preemption count were zero in all rows.

Report takeaway: the current hardware sweep validates the end-to-end
Modal-to-vLLM measurement loop and gives real TPOT/TTFT scale for Qwen2.5-1.5B.
It does not yet validate KV-blocking behavior, because even the lowest realized
KV cache was large enough for the 300-request workload and vLLM queued work
rather than rejecting it.

## SimPy vs vLLM Interpretation

The SimPy and vLLM sweeps should not be plotted as a strict one-to-one
calibration yet:

- SimPy uses synthetic lognormal lengths and simplified iteration latency.
- vLLM uses real ShareGPT prompts and real A10G serving behavior.
- The SimPy sweep currently covers smaller B/K regimes than the vLLM sweep.
- vLLM does not expose blocking in the same way as the SimPy admission model;
  it queues and may preempt instead of immediately rejecting at arrival.

The fair comparison for the report is qualitative:

1. Both systems show goodput increasing with offered load.
2. vLLM's measured latency scale is much lower than the original toy SimPy
   constants, motivating the measured `results/latency_curve.json` calibration.
3. The analytical model should be calibrated to measured service rates before
   claiming quantitative agreement with hardware.
4. KV blocking remains a follow-up measurement target; the current hardware
   grid did not reach a blocking/preemption regime.

## Suggested Final Report Claims

- We built a reusable SimPy simulator for continuous batching with finite KV
  budget, Poisson arrivals, prompt/generation lengths, TTFT, goodput, and
  blocking metrics.
- We implemented a closed-form analytical approximation and comparison harness.
  It predicts goodput reasonably well but needs a richer tail model for p95/p99
  TTFT under overload.
- We created a canonical ShareGPT trace so simulation and vLLM measurements can
  replay the same request-length pool.
- We measured vLLM on Modal/A10G with Qwen2.5-1.5B-Instruct. The measured sweep
  completed 64 benchmark rows and showed mean TTFT below 61 ms, p99 TTFT below
  185 ms, and TPOT around 8 to 10 ms/token.
- The current hardware grid did not produce blocking or preemption; the next
  experiment should either increase request pressure/concurrency, reduce
  effective memory, or use longer traces to force KV contention.

## Suggested Limitations Paragraph

The simulator and analytical model are intentionally simplified. They model a
single GPU and a single model, assume Poisson arrivals, and approximate vLLM's
continuous batching at an iteration level. The first analytical baseline uses
average request sizes and an approximate tail distribution, so it is less
accurate for p95/p99 TTFT in overload regimes. The vLLM measurements validate
the empirical pipeline and provide real latency scale, but the measured grid did
not yet reach a KV-blocking regime; therefore, hardware blocking validation
remains future work.
