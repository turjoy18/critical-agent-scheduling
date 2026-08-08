# KVFlow conceptual reproduction (Layers 1–2)

Pure-Python stand-in for [KVFlow](https://arxiv.org/abs/2507.07400)’s Agent Step Graph and
workflow-aware prefix-cache policies. No SGLang, no real KV tensors.

This package optimizes **cache lifetime** (what prefixes to keep / prefetch). The rest of
this repo profiles **compute priority** (which agent should run when). Same research/writing
DAG, different control knob.

For the written discussion (tables, figures, takeaways), see
[`docs/REPORT.md`](../docs/REPORT.md) §9.

## Layout

| Module | Role |
|---|---|
| `step_graph.py` | Agent Step Graph + `steps_to_execution` (STE) |
| `workflows.py` | Research/writing DAG + OR-branch toy + sequential chain |
| `compare_criticality.py` | STE vs runtime-weighted critical-path frequency |
| `cache_sim.py` | Prefix cache simulator (LRU / STE / STE+prefetch) |
| `experiment.py` | Synthetic workload harness + CSV/plots |
| `tests/` | Unit tests |

## Run

```bash
# Layer 1 — STE vs critical path on existing traces
python -m kvflow.compare_criticality --traces simulated/traces.jsonl

# Layer 2 — cache policy comparison (default: sequential 10-agent chain)
# Note: no spaces after commas in --configs
python -m kvflow.experiment --configs lru,ste,ste+prefetch --out kvflow/results

# Same policies on the research/writing DAG
python -m kvflow.experiment --graph research --out kvflow/results

# Tests
python -m unittest discover -s kvflow/tests -v
```

Artifacts in `kvflow/results/`:

| File | Contents |
|---|---|
| `ste_vs_critical_path.csv` / `.png` | Layer 1 table + dual-axis plot |
| `summary.csv` | Layer 2 metrics per setting × policy |
| `cache_policy_comparison.png` | Latency, hit rate, stall, evictions |

## Results snapshot (for report)

**Layer 1 (20 simulated runs):** researchers A/B/C all have STE = 1, but only B has
critical-path frequency 1.0 (slowest parallel peer). STE ≠ compute criticality.

**Layer 2 (`large`, sequential:10, after warm-up):**

| Policy | Avg latency (s) | Hit rate | Stall (s) |
|---|---:|---:|---:|
| lru | 34.06 | 0.00 | 12.34 |
| ste | 33.85 | 0.10 | 11.11 |
| ste+prefetch | 32.21 | 0.90 | 1.28 |

Full small/medium/large table: `results/summary.csv` or REPORT §9.2.

## Notes

- STE is **structure-only** (how soon an agent runs). Critical-path frequency is
  **latency-weighted** from measured runtimes — they need not agree.
- Cache timings use named constants in `cache_sim.py` for reproducibility, not hardware parity.
- Workload sizes `small` / `medium` / `large` use tight capacity (shared + two fixed leaves + one dynamic).
- Default experiment graph is `sequential:10` (clear STE/prefetch signal). Use `--graph research` for the profiling DAG.
