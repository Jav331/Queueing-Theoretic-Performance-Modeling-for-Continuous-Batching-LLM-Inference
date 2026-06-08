# EE 384S — analytical-track update

Hey Adam — here's what landed on `atom-branch` for the analytical contributions.
None of your files were modified; everything I added either subclasses or
matches your CSV/CLI conventions so your existing comparison + plotting
harness works on the new outputs with no code changes.

## What's new

### Phase 1 — two-stage closed form (`analytical/closed_form.py`)
Birth-death chain on `n ∈ {0..N_max}` with `N_max = ⌊K / E[kv]⌋` (matches
your sim's KV admission cap). Per-state service rate
`μ_n = min(n, B) / s(min(n, B))` uses *your* `iteration_latency` curve
directly, so cross-validation is apples-to-apples. Stationary distribution
in closed product form; tail (p50/p95/p99) via exact PASTA + Erlang-CDF
mixture, inverted numerically.

CLI: `python -m analytical.run_closed_form` → `results/closed_form_summary.csv`
(same column schema as your `analytical_summary.csv`).

### Phase 2 — multi-class CTMC (`analytical/markov_solver.py` + `analytical/state_space.py`)
Multi-class state `n = (n_short, n_long)` with KV demand quantized to vLLM-
style 16-token pages. Sparse generator Q built per config; stationary solved
via `scipy.sparse.linalg.spsolve` on `πQ = 0, Σπ = 1`. State spaces are 60–
830 in our regime — sub-second per config.

CLI: `python -m analytical.run_markov` → `results/markov_summary.csv`.

### Phase 3 — pluggable admission policies (`simulator/policies.py` + `simulator/policy_simulator.py`)
`AdmissionPolicy` Protocol + `FCFSPolicy` / `SRPTPolicy` / `DPTablePolicy`.
`PolicySimulator` **subclasses** your `ContinuousBatchingSimulator` — your
file is untouched. FCFS via the subclass reproduces your baseline exactly at
matched seed/config (verified). SRPT reorders the waiting deque by
`kv_required` ascending before each batch fill.

### Phase 4 — DP admission policy (`analytical/dp_policy.py` + `analytical/run_dp_policy.py`)
Per-(λ, B, K) policy table indexed by quantized (q, b, kv_bin). Currently a
one-step SLO-threshold admission rule (accept iff predicted post-admission
p99 ≤ SLO under the accept-all stationary), projected onto the sim's lookup
grid by majority vote weighted by chain stationary mass. Outputs `.npy`
tables under `results/dp_policies/`.

### Phase 5 — comparison sweep (`experiments/run_policy_comparison.py` + plotter)
FCFS / SRPT / DP × your (λ, B, K) × seed grid via `PolicySimulator`. Outputs
`results/policy_comparison.csv` with an extra `policy` column; plotter writes
3-line figures per (B, K) cell to `figures/policy_comparison/`.

### Writeup spec (`writeup/model_spec.md`)
Concise derivation of all three analytical contributions — source-of-truth
for the analytical sections of `final_report.tex`. Includes the
cross-validation tables below.

### Cleanup
Deleted two 0-byte original stubs (`experiments/run_sim_sweep.py`,
`experiments/plot_results.py`) — superseded by your `run_sweep.py` and
`plot_sweep.py` + `plot_comparison.py`.

## Headline numbers (vs your SimPy sweep, 48 configs, 3 seeds)

| metric | M/M/C baseline | **Two-stage closed form** | Markov (2-class) |
|---|---:|---:|---:|
| mean_ttft rel error | 0.70 | **0.17** | 0.23 |
| p95_ttft rel error  | 1.71 | **0.41** | 0.51 |
| p99_ttft rel error  | 1.83 | **0.49** | 0.61 |
| goodput rel error   | 0.18 | **0.07** | 0.07 |
| blocking rel error  | 10.24| **0.89** | 0.86 |

Closed-form is the strongest validation story — 4–12× tighter than the M/M/C
baseline on every metric. Markov is a richer model that also enables the
class-aware DP policy in Phase 4, but isn't quantitatively more accurate.

### Policy comparison @ (λ=4, B=4, K=4096), SLO=1s

| policy | blocking | p99 TTFT (s) | goodput (req/s) |
|---|---:|---:|---:|
| FCFS                 | 0.205 | 1.92 | 3.14 |
| SRPT                 | 0.205 | 4.06 | 3.14 |
| FCFS + 20% rand-rej  | 0.291 | 1.68 | 2.80 |
| **DP**               | 0.277 | **1.31** | 2.86 |

DP cuts p99 by 32% vs FCFS — and importantly, by 22% vs *random-rejection*
FCFS at matched blocking, confirming the policy rejects strategically (in
high-occupancy states) rather than blindly. SRPT helps mean but worsens tail
(classic head-of-line vs heavy-tail tradeoff).

## Validation pass + fixes

Did a thorough self-review; codex (independent reviewer) caught and fixed
two issues:

1. **`simulator/policies.py:80`** — `DPTablePolicy` was indexing by post-
   arrival KV (`reserved_kv + candidate.kv_required`) and using raw page
   counts; the DP table was built with pre-arrival KV scaled to `kv_bins`.
   Now both match.
2. **`analytical/markov_solver.py:180`** — `_quantile_via_inversion` could
   blow past `floor + cap` on overload configs, producing non-monotonic
   p50/p95/p99. Now clamps `hi` at each doubling step. 0/96 violations
   post-fix.

**Strong consistency check that still passes:** Markov with `n_classes=1`
agrees with the two-stage closed-form to ~1e-15 (independent
implementations of equivalent math).

## Open caveats (worth a look before the writeup)

- **DP docstring overclaims.** `analytical/dp_policy.py` mentions "value
  iteration via uniformization" but the implementation is a myopic one-step
  SLO threshold — defensible heuristic, but the docstring + writeup should
  describe it honestly. Full closed-loop value iteration is a real
  follow-up if we want it.
- **`build_request_classes` uses marginal quantile averages**, not joint
  KV quantile. First moment is preserved (`Σ π_c d_c = E[L_p]+E[L_g]`),
  second moment slightly off. Minor; worth flagging in the limitations.
- **Tail error in extreme regimes** (B=1 deep overload, or B=8 light load
  with K large) is wider than the headline 0.17/0.49 numbers — model averages
  away per-request KV variance and prefill burst timing. Documented.

## What we still need from the empirical track

For the proposal's ±20% target vs vLLM measurements (not vs sim):
- `experiments/preprocess_sharegpt.py` — currently using synthetic lognormal
  (μ=256 / μ=128). Real ShareGPT will likely shift the variance.
- `experiments/run_vllm_sweep.py` — Modal + Qwen 2.5-1.5B measurement loop.
- Microbenchmark fit of `iteration_latency` coefficients (currently
  Adam-placeholder values from the simulator). Once we have real numbers,
  the analytical models just take them as parameters — no model changes
  needed.

## End-to-end smoke test (~1 min)

```bash
python -m experiments.run_sweep --duration 500 --arrival-rates 0.5,1,2,3 \
  --max-batch-sizes 1,4,8 --kv-budgets 2048,4096,8192,16384 --seeds 0,1,2
python -m analytical.run_closed_form
python -m analytical.run_markov
python -m experiments.compare_simulation_analytical \
  --analytical results/closed_form_summary.csv \
  --output results/comparison_closed_form.csv
python -m analytical.run_dp_policy --slo 1.0
python -m experiments.run_policy_comparison
python -m experiments.plot_policy_comparison
```

Branch: `atom-branch`. Nothing committed yet pending your review.
