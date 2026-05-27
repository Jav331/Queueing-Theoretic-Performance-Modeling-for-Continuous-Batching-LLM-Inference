# EE384S-Project
Queueing-Theoretic Performance Model for Continuous-Batching LLM Inference

## Baseline SimPy simulator

Run the baseline continuous-batching simulator from the repository root:

```powershell
.\.venv\Scripts\python.exe -m simulator.run_baseline --duration 200 --arrival-rate 2 --max-batch-size 8 --kv-budget 8192
```

The command writes per-request results to `results/baseline.csv` and prints TTFT percentiles, mean TTFT, goodput, and blocking probability.
