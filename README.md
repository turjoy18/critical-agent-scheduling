# Critical-Agent Scheduling for Multi-Agent LLM Systems

Empirical profiling and early systems work on **which agents dominate end-to-end latency** in multi-agent LLM workflows, plus a conceptual reproduction of **KVFlow**-style steps-to-execution (STE) and prefix-cache policies.

> Same workflow DAG, two control knobs: **compute priority** (who runs when) vs **cache lifetime** (which KV prefixes to keep / prefetch).

## Motivation

Multi-agent systems often run planner → parallel researchers → join → critic → writer. Latency is usually dominated by a small set of *critical* agents on the longest dependency path—not by every agent equally. This repo validates that empirically, then explores related cache-aware ideas from KVFlow.

## Repository layout

| Path | Purpose |
|------|---------|
| `simulated/` | Sleep-based LangGraph agents, traces, critical-path analysis |
| `llm/` | OpenRouter LLM agents, traces, plots |
| `kvflow/` | Pure-Python Agent Step Graph + LRU / STE / STE+prefetch cache sim |
| `docs/REPORT.md` | Initial profiling report (local; may be gitignored) |
| `requirements.txt` | Python dependencies |

## Quick start

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
cp env.example .env             # add OPENROUTER_API_KEY for llm/
```

### Profiling (compute criticality)

```bash
cd simulated && python workflow.py && python analyze.py
cd ../llm && python workflow_llm.py && python analyze_llm.py
```

### KVFlow Layers 1–2 (conceptual; no GPU)

```bash
python -m kvflow.compare_criticality --traces simulated/traces.jsonl
python -m kvflow.experiment --configs lru,ste,ste+prefetch --out kvflow/results
python -m unittest discover -s kvflow/tests -v
```

See [`kvflow/README.md`](kvflow/README.md) for details.

## Workflow

```text
planner
  → researcher_a / researcher_b / researcher_c   (parallel)
  → synthesizer
  → critic
  → final_writer
```

Critical path is the **runtime-weighted** longest path over this DAG. In profiling, parallel researchers share the same structural depth, but the slowest peer (e.g. researcher_b) dominates critical-path frequency.

## KVFlow snapshot

- **Layer 1:** STE (structure-only) vs critical-path frequency — they need not agree.
- **Layer 2:** Under tight prefix-cache capacity, **STE+prefetch** beats LRU on hit rate and stall (conceptual timings; not hardware parity).

## Status

- [x] Simulated + LLM critical-path profiling
- [x] KVFlow STE + cache-policy conceptual sim
- [ ] Priority-aware ready-queue scheduling (FIFO vs critical-path-first / HEFT-style)
- [ ] Joint compute + cache scheduling under contention

## References

- Pan et al. 2025. *KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows.* [arXiv:2507.07400](https://arxiv.org/abs/2507.07400)
- Topcuoglu et al. 2002. HEFT scheduling. IEEE TPDS.
- LangGraph — graph-based agent workflows
