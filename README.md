# Queueing-Theoretic Performance Modeling for Continuous-Batching LLM Inference

EE384S final project repository for modeling and measuring continuous-batching
LLM inference under finite KV-cache capacity.

The project combines three tracks:

- a SimPy simulator for queueing-level continuous batching,
- analytical approximations for TTFT, goodput, and blocking,
- Modal/vLLM measurements on `Qwen/Qwen2.5-1.5B-Instruct`.

The main question is how arrival rate, batch width, request length, and KV-cache
capacity affect time to first token (TTFT), tail latency, goodput, and blocking
or preemption behavior.

## Repository Structure

| path | purpose |
|---|---|
| `simulator/` | baseline SimPy continuous-batching simulator and metrics |
| `analytical/` | closed-form, Markov, and measured-service-rate analytical models |
| `experiments/` | sweeps, plotting, ShareGPT preprocessing, Modal/vLLM harnesses |
| `infra/` | small Modal/vLLM smoke app |
| `data/` | canonical replay traces and empirical length distributions |
| `results/` | CSV/JSON outputs from simulator, analytical, and vLLM runs |
| `figures/` | generated report figures |
| `writeup/` | model notes and report-ready result summaries |

## Key Results So Far

### Analytical vs SimPy

The analytical baseline was compared against the SimPy sweep over 48 matched
configurations. The strongest agreement is on goodput; p95/p99 TTFT remain the
hardest metrics because the baseline uses a simplified tail approximation.

| metric | mean relative error | median relative error |
|---|---:|---:|
| mean TTFT | 0.712 | 0.325 |
| p95 TTFT | 1.821 | 0.295 |
| p99 TTFT | 1.894 | 0.453 |
| goodput | 0.177 | 0.106 |
| blocking probability | 4.410 | 0.056 |

### Modal/vLLM Sweep

The vLLM sweep completed 64 rows on Modal/A10G:

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- vLLM version: `0.22.0`
- arrival rates: `0.5,1,2,3,4,5,6,8` req/s
- max sequence counts: `64,128`
- GPU memory utilization: `0.3,0.5,0.7,0.9`
- realized KV cache: `104,336` to `614,736` tokens
- benchmark size: 300 measured requests plus 8 warmup requests per row

Headline hardware measurements:

- max observed goodput: `6.74 req/s`
- worst observed p99 TTFT: `0.185 s`
- mean TTFT stayed below `0.061 s`
- mean TPOT ranged from about `8.3` to `10.2 ms/token`
- blocking probability and preemption count were zero in this measured grid

The current vLLM sweep validates the full measurement pipeline and provides
real latency scale, but it does not yet validate KV-blocking behavior. The next
hardware experiment should increase effective pressure or request lengths until
vLLM begins queueing deeply, preempting, or failing requests.

More report-ready language is in `writeup/report_results.md`.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The Modal experiments require a Modal token and the intended profile. For this
project we used:

```powershell
$env:MODAL_PROFILE='adamco27'
```

## Baseline SimPy Simulator

Run one baseline simulation:

```powershell
.\.venv\Scripts\python.exe -m simulator.run_baseline --duration 200 --arrival-rate 2 --max-batch-size 8 --kv-budget 8192
```

The command writes per-request results to `results/baseline.csv` and prints
TTFT percentiles, mean TTFT, goodput, and blocking probability.

TTFT percentiles are computed over accepted requests that complete before the
simulation ends. Rejected requests and accepted requests still in service at the
horizon are excluded from TTFT percentile calculations.

## SimPy Sweep and Plots

Run the default simulator sweep:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_sweep
```

The default sweep uses duration `1000`, arrival rates `0.5,1,2,3,4,5,6,8`,
max batch sizes `1,2,4,8,16`, KV budgets `2048,4096,8192,16384`, and seeds
`0,1,2,3,4`. It writes `results/sweep_summary.csv`.

Generate report figures:

```powershell
.\.venv\Scripts\python.exe -m experiments.plot_sweep
```

The KV-budget plots hold max batch size fixed at `B=8`. As KV budget increases,
the simulator may admit more requests; this can reduce blocking while
increasing effective load and tail latency, so p99 TTFT can rise even though
fewer requests are rejected.

## Analytical Models

Run the first analytical approximation:

```powershell
.\.venv\Scripts\python.exe -m analytical.run_analytical
```

Compare analytical predictions against SimPy:

```powershell
.\.venv\Scripts\python.exe -m experiments.compare_simulation_analytical
```

Generate comparison plots:

```powershell
.\.venv\Scripts\python.exe -m experiments.plot_comparison
```

The repository also includes richer closed-form and Markov-chain analytical
models, plus optional measured-service-rate injection from
`results/latency_curve.json`.

## ShareGPT Trace

Build the canonical replay trace:

```powershell
.\.venv\Scripts\python.exe -m experiments.preprocess_sharegpt --sample-size 5000 --seed 0 --max-model-len 2048
```

This writes:

- `data/sharegpt_trace.parquet`
- `data/length_dist.json`

The trace is the single source of truth for vLLM replay and future
trace-driven simulation.

## Modal/vLLM Microbenchmark

Measure decode step latency on Modal:

```powershell
$env:MODAL_PROFILE='adamco27'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
.\.venv\Scripts\modal.exe run experiments/microbench_latency.py --output results/latency_curve.json
```

This boots vLLM, measures fixed active batch sizes, fits a sublinear decode
latency curve, and writes `results/latency_curve.json`.

## Modal/vLLM Serving Sweep

Preview the full grid without launching a GPU job:

```powershell
.\.venv\Scripts\python.exe -m experiments.run_vllm_sweep --dry-run
```

Run a small smoke test:

```powershell
$env:MODAL_PROFILE='adamco27'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
.\.venv\Scripts\modal.exe run experiments/run_vllm_sweep.py --smoke
```

Run the full vLLM sweep:

```powershell
$env:MODAL_PROFILE='adamco27'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
.\.venv\Scripts\modal.exe run experiments/run_vllm_sweep.py
```

The harness replays `data/sharegpt_trace.parquet` by converting it to vLLM's
custom JSONL benchmark format, preserving `prompt_text` and forcing per-request
output length from `gen_len`. It writes:

- `results/vllm_sweep_summary.csv`
- `results/vllm_sweep_metadata.json`

`results/vllm_sweep_summary.csv` intentionally keeps the same column schema as
`results/sweep_summary.csv`. vLLM version, flags, TPOT, throughput, preemption
count, and realized KV cache metadata are recorded in the metadata JSON.

## Report Notes

Use these files when writing the final report:

- `writeup/report_results.md`: current result summary and suggested claims
- `writeup/model_spec.md`: analytical model derivations and assumptions
- `writeup/notes_for_adam.md`: analytical-track implementation notes

## Project Naming

Project title: **Queueing-Theoretic Performance Modeling for Continuous-Batching
LLM Inference**.

Suggested GitHub repository name if renaming the remote:

```text
queueing-llm-inference-modeling
```
