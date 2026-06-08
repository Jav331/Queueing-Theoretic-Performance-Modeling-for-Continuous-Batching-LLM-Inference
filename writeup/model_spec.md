# Analytical-Track Model Specification

Source-of-truth for the queueing models behind the EE 384S project's three
analytical contributions. The corresponding sections of `final_report.tex`
will be a polished/condensed version of this document.

## 1. Shared assumptions

We model a single-GPU, single-model continuous-batching engine (vLLM-style).
Arrivals are Poisson at rate λ. Each request carries a bivariate service
demand (prompt length L_p, generation length L_g) drawn from lognormal
distributions (medians 256 and 128, σ = 0.55 and 0.70). KV cache is a
finite pool of K tokens; an admitted request reserves L_p + L_g tokens
until completion.

The per-iteration latency curve is workload-dependent. For consistency with
the simulator (`simulator.simpy_simulator.iteration_latency`) we use

  prefill(b, T) = 0.004 + 8e-5 · T · f(b)
  decode(b)     = 0.006 + 5.5e-4 · f(b)
  f(b)          = 1 + 0.22 · (b − 1)^0.65

with active batch size b and prompt-token count T. Service time per request
at batch b is s(b) = prefill(b, E[L_p]) + E[L_g] · decode(b).

All three contributions reuse `iteration_latency` directly so analytical
predictions and simulator measurements share the same physical service
mechanism — this is what makes the cross-validation interpretable.

## 2. Contribution #1 — Two-stage closed-form (`analytical/closed_form.py`)

We decompose the engine into two stages coupled through the KV constraint:

  Stage A — admission buffer (requests waiting for an active batch slot).
  Stage B — prefill/decode PS pool (up to B concurrent jobs sharing GPU).

Let n denote total requests in system. Active count is min(n, C) where
C = min(B, ⌊K / E[kv]⌋) and E[kv] = E[L_p] + E[L_g]. The KV-implied
admission cap is N_max = ⌊K / E[kv]⌋: any arrival to state N_max is lost
(matches the simulator's reserved_kv test).

The system is a birth-death chain on {0, …, N_max} with state-dependent
service rate

  μ_n = min(n, C) / s(min(n, C)).

Stationary distribution in product form:

  p_n = p_0 ∏_{k=1..n} λ / μ_k,   p_0 = (Σ_n ∏)^{-1}.

Metrics:
- Blocking probability: P_block = p_{N_max}.
- Goodput: λ_eff = λ (1 − P_block).
- Mean TTFT: E[W_q] + s_{ft}(b̄), where E[W_q] = L_q / λ_eff via Little's
  law on the waiting queue, and s_{ft}(b̄) = prefill(b̄, E[L_p]) +
  decode(b̄) at the typical busy-system active batch b̄.
- Tail TTFT (p_q quantile): by PASTA, an accepted arrival sees pre-arrival
  state n with probability p_n / (1 − P_block). Conditional on n ≥ C, its
  wait time is Erlang((n + 1 − C), μ_C) with μ_C = C/s(C); conditional on
  n < C, wait = 0. The TTFT CDF is the resulting mixture, and we invert it
  numerically (binary search) to read off p50/p95/p99.

**Cross-validation result.** Averaged over 48 configs (λ ∈ {0.5, 1, 2, 3},
B ∈ {1, 4, 8}, K ∈ {2048, 4096, 8192, 16384}, 3 seeds), relative error vs
the SimPy sim:

| metric     | M/M/C (Adam baseline) | Two-stage closed form |
|-----------:|----------------------:|----------------------:|
| mean_ttft  | 0.70                  | **0.17**              |
| p95_ttft   | 1.71                  | **0.41**              |
| p99_ttft   | 1.83                  | **0.49**              |
| goodput    | 0.18                  | **0.07**              |
| blocking   | 10.24                 | **0.89**              |

The two-stage chain dominates on every metric — most strikingly on
blocking (the M/M/C Erlang-B formula misses heavy overload by an order of
magnitude; the chain captures KV saturation exactly).

## 3. Contribution #2 — Multi-class Markov chain (`analytical/markov_solver.py`)

The 1-D chain above collapses every request to E[kv]. The proposal calls
for a Markov chain over (queue length, active batch, aggregate KV
occupancy); the (q, b, kv) triple is only informative beyond (q, b) when
KV varies per request. We model that by splitting the lognormal joint
demand into K equal-probability size classes (default K = 2: short and
long), each with its own KV demand d_c in pages (page size 16 tokens,
matching vLLM PagedAttention).

State: n = (n_1, …, n_K), with n_c the number of class-c requests in
system. Feasible set: {n : Σ_c n_c · d_c ≤ K/page}. Page-granular
discretization is the "tractability lever" the proposal anticipated —
state-space size grows linearly with K rather than as K · B · K_pages.

Generator Q (with arrivals λ_c = λ · π_c per class, proportional-sharing
completions):

  Arrival class c:   n → n + e_c   rate λ_c if feasible
  Class-c complete:  n → n − e_c   rate (active(n) / s(active(n))) · (n_c / Σn)
  where active(n) = min(Σn, B).

Solve π Q = 0, Σ π = 1 by sparse linear solve (`scipy.sparse.linalg`).
State spaces in our regime: 61 (K=4096, 2 classes) up to 827 (K=16384, 2
classes); growing modestly with K.

Metrics derived from π as above, but with per-class arrival weights.

**Cross-validation result.** Same 48-config sweep:

| metric     | Two-stage closed form | Multi-class Markov (2 cls) |
|-----------:|----------------------:|---------------------------:|
| mean_ttft  | 0.17                  | 0.23                       |
| p95_ttft   | 0.41                  | 0.51                       |
| p99_ttft   | 0.49                  | 0.61                       |
| goodput    | 0.07                  | 0.07                       |
| blocking   | 0.89                  | 0.86                       |

The 2-class Markov chain is comparable to the closed form (within ~35 % on
tail). It does NOT trivially beat the closed form because (a) the 2-class
split captures less variance than the underlying lognormal, and (b) the
proportional-sharing completion approximation introduces noise. The
Markov chain's primary value-add is enabling the per-state DP admission
policy in contribution #3 — the closed-form 1-D state isn't expressive
enough to underwrite class-aware accept/reject.

## 4. Contribution #3 — DP admission policy (`analytical/dp_policy.py`)

Cast admission as an MDP on the Phase-2 state space:

  State:    n = (n_1, ..., n_K)
  Action:   on each arrival, ACCEPT or REJECT
  Reward:   +1 if predicted post-admission p99 ≤ SLO; −penalty otherwise
            (default penalty = 1)
  Reject reward: 0

We solve a one-step expected-reward maximization per state, using the
chain's predicted per-state p99 TTFT (from the Erlang mixture above) as
the SLO signal. Output: a per-state ACCEPT bit, then projected onto the
quantized (q, b, kv_bin) lookup grid the simulator's `DPTablePolicy`
consumes. Each (λ, B, K) cell gets its own `.npy` table.

**Comparison sweep (`experiments/run_policy_comparison.py`)** runs FCFS,
SRPT, and DP through the same simulator across (λ, B, K, seed):

  B=4, K=4096, λ=4:  FCFS p99 = 1.92, DP p99 = 1.31  (DP −32 %)
                     FCFS goodput = 3.14, DP goodput = 2.86  (DP −9 %)
                     FCFS blocking = 21 %, DP blocking = 28 %

  B=4, K=4096, λ=3:  FCFS p99 = 1.59, DP p99 = 1.25  (DP −22 %)

DP successfully trades a small goodput hit for a clear tail-latency win
under overload — the canonical SLO-aware admission control behavior. SRPT
helps mean TTFT but worsens p99 (classic head-of-line vs heavy-tail
tradeoff), confirming the proposal's intuition that pure SRPT is not the
right policy under SLO constraints.

## 5. Scope, limitations, future work

The model space currently in scope (matches the proposal's restrictions):
- single-model, single-GPU
- iteration-level continuous batching with paged KV
- Poisson arrivals (heavy-tailed extension noted as TODO)
- proportional-sharing across active batch for class completion rates

Explicit out-of-scope (will be limitations in the writeup):
- tensor parallelism
- chunked prefill (Sarathi-Serve)
- speculative decoding
- prefill/decode disaggregation (DistServe)

Future refinement directions (not promised in the proposal):
- Phase-type approximation for the wait distribution (more accurate tails)
- Per-class queue tracking (replaces proportional sharing with exact FCFS)
- Calibrate iteration-latency constants from real microbenchmarks once
  Modal/vLLM measurements land (Adam's empirical track, W3).

## 6. Files and how to run

```
analytical/
  closed_form.py        # contribution #1: two-stage birth-death chain
  run_closed_form.py    # CLI → results/closed_form_summary.csv
  markov_solver.py      # contribution #2: multi-class CTMC
  state_space.py        # state enumeration + KV-feasibility
  run_markov.py         # CLI → results/markov_summary.csv
  dp_policy.py          # contribution #3: MDP value-iteration
  run_dp_policy.py      # CLI → results/dp_policies/*.npy
  approximation.py      # Adam's M/M/C baseline (kept as reference)
  run_analytical.py     # CLI for the M/M/C baseline

simulator/
  policy_simulator.py   # subclass of Adam's sim; injects admission policy
  policies.py           # FCFSPolicy, SRPTPolicy, DPTablePolicy
  simpy_simulator.py    # Adam's baseline (UNTOUCHED)
  metrics.py            # Adam (UNTOUCHED)
  run_baseline.py       # Adam (UNTOUCHED)

experiments/
  run_policy_comparison.py   # FCFS/SRPT/DP sweep via PolicySimulator
  plot_policy_comparison.py  # 3-policy lines per (B, K) cell
  compare_simulation_analytical.py  # Adam (UNTOUCHED; works on any of
                                    # analytical_summary, closed_form_summary,
                                    # markov_summary by --analytical arg)
  plot_comparison.py    # Adam (UNTOUCHED)
  run_sweep.py          # Adam (UNTOUCHED)
  plot_sweep.py         # Adam (UNTOUCHED)
```

End-to-end smoke test:

```bash
python -m experiments.run_sweep --duration 500 --arrival-rates 0.5,1,2,3 \
  --max-batch-sizes 1,4,8 --kv-budgets 2048,4096,8192,16384 --seeds 0,1,2
python -m analytical.run_closed_form
python -m analytical.run_markov
python -m experiments.compare_simulation_analytical \
  --analytical results/closed_form_summary.csv \
  --output results/comparison_closed_form.csv
python -m experiments.compare_simulation_analytical \
  --analytical results/markov_summary.csv \
  --output results/comparison_markov.csv
python -m analytical.run_dp_policy --slo 1.0
python -m experiments.run_policy_comparison
python -m experiments.plot_policy_comparison
```
