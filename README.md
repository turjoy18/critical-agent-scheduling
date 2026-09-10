# Critical-Agent Scheduling for Multi-Agent LLM Systems

Empirical profiling and early systems work on **which agents dominate end-to-end latency** in multi-agent LLM workflows, plus a modular discrete-event **scheduler** (FIFO, Critical-Path-First, Least-Slack-First, HEFT) and a conceptual reproduction of **KVFlow** prefix-cache policies.

> Same workflow DAG, two control knobs: **compute priority** (who runs when) vs **cache lifetime** (which KV prefixes to keep / prefetch).

## Motivation

Multi-agent systems often run planner → parallel researchers → join → critic → writer. Latency is usually dominated by a small set of *critical* agents on the longest dependency path—not by every agent equally. This repository validates that empirically, implements priority-aware scheduling algorithms under resource contention, and explores cache-aware prefix policies.

## Repository layout

| Path               | Purpose                                                           |
| ------------------ | ----------------------------------------------------------------- |
| `simulated/`       | Sleep-based LangGraph agents, traces, critical-path analysis      |
| `llm/`             | OpenRouter LLM agents, traces, plots                              |
| `kvflow/`          | Pure-Python Agent Step Graph + LRU / STE / STE+prefetch cache sim |
| `scheduler/`       | Discrete-event scheduler (FIFO, CPF, Slack, HEFT) + live Groq API |
| `docs/REPORT.md`   | Initial empirical profiling report                                |
| `docs/SCHEDULING_REPORT.md` | Policy evaluation & benchmark report                     |
| `requirements.txt` | Python dependencies                                               |

## Quick start

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
cp env.example .env             # add OPENROUTER_API_KEY and/or GROQ_API_KEY
```

### 1. Profiling (compute criticality)

```bash
cd simulated && python workflow.py && python analyze.py
cd ../llm && python workflow_llm.py && python analyze_llm.py
```

### 2. Scheduling Policies (FIFO, CPF, Slack, HEFT)

The discrete-event simulator loads **agent-level mean runtimes** from `llm/profiling_summary_llm.csv` (and `simulated/profiling_summary.csv` for the sleep profile)—not per-run trace replay. See [`docs/SCHEDULING_REPORT.md`](docs/SCHEDULING_REPORT.md) §3.

```bash
# Run unit tests
python -m unittest discover -s scheduler/tests -v

# Run simulation sweeps (worker slots, contention, batch scaling, noise)
python -m scheduler.experiment --out scheduler/results

# (Optional) Run live Groq LLM validation
python -m scheduler.live_runner --runs 5 --workers 2
```

See [`docs/SCHEDULING_REPORT.md`](docs/SCHEDULING_REPORT.md) for detailed evaluation results and plots.

### 3. KVFlow Layers 1–2 (conceptual; no GPU)

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

## Status

- [x] Simulated + LLM critical-path profiling
- [x] KVFlow STE + cache-policy conceptual sim
- [x] Priority-aware ready-queue scheduling (FIFO vs Critical-Path-First vs Slack vs HEFT)
- [ ] Joint compute + cache scheduling under contention

## References

- Pan et al. 2025. *KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows.* arXiv:2507.07400.
- Pearson et al. 2026. *Graph-Based Agentic AI with LangGraph: Workflow Pathways for Long-Running Stateful Business Processes.* arXiv:2607.19297.
- H. Topcuoglu, S. Hariri and Min-You Wu, "Performance-effective and low-complexity task scheduling for heterogeneous computing," in IEEE Transactions on Parallel and Distributed Systems, vol. 13, no. 3, pp. 260-274, March 2002.
- Wu, Q. et al. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. arXiv:2308.08155.
