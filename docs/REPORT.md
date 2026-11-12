# Critical-Agent-Aware Scheduling for Multi-Agent LLM Systems
## Initial Profiling Report

---

## 1. Introduction and Motivation

Multi-agent LLM systems divide complex tasks among specialized agents. End-to-end latency is often dominated by a small set of *critical agents* on the longest dependency path, rather than by all agents equally.

This report presents an initial empirical study that:

1. Builds a multi-agent workflow in LangGraph
2. Collects per-agent execution traces
3. Computes critical paths over the agent dependency graph
4. Checks whether some agents consistently contribute disproportionately to end-to-end latency

This addresses supervisor feedback to validate the motivation empirically before designing a new scheduling policy.

---

## 2. Research Question

> Do multi-agent LLM workflows show stable and meaningful differences in agent criticality across repeated executions?

If yes, critical-agent-aware scheduling is justified.

---

## 3. System Design

### 3.1 Workflow Structure

The prototype uses a fixed DAG-style LangGraph workflow with both sequential and parallel stages:

```text
START
  → planner
  → researcher_a / researcher_b / researcher_c   (parallel)
  → synthesizer   (join: waits for all researchers)
  → critic
  → final_writer
  → END
```

**Figure 1.** Workflow dependency graph. Red nodes/edges mark the most common critical path from LLM runs.

![Workflow dependency graph](llm/workflow_graph_llm.png)

### 3.2 Instrumentation

Each agent call records:

- agent name
- parent dependencies
- start/end timestamps
- runtime (seconds)
- input/output tokens (LLM runs)

Traces are stored in JSONL format and analyzed offline.

### 3.3 Critical-Path Method

Agents are modeled as a weighted DAG. Node weight = measured runtime. The critical path is the longest path from start to end:

- Compute earliest finish times in topological order
- Backtrack parent pointers to recover the critical path
- Aggregate critical-path frequency across runs

Waiting time is also measured as:

```text
waiting_time = max(0, actual_start - max(parent_end_times))
```

---

## 4. Experimental Setup

### 4.1 Simulated Baseline

| Item | Value |
|---|---|
| Framework | LangGraph |
| Agents | Simulated with `time.sleep()` |
| Runs | 20 |
| Purpose | Validate tracing and critical-path code |

`researcher_b` was intentionally slower to test whether critical-path detection works.

### 4.2 Real LLM Experiment

| Item | Value |
|---|---|
| Framework | LangGraph |
| Model | `openai/gpt-oss-20b:free` via OpenRouter |
| Runs | 5 |
| Tasks | KV caching; LangGraph vs AutoGen; DAG scheduling; critical path analysis; HEFT |

Prompt asymmetry: Researcher B was given a heavier analysis prompt (longer expected output) than A/C, reflecting realistic uneven agent workloads.

---

## 5. Results

### 5.1 Simulated Results (20 runs)

Average end-to-end critical-path latency: **~6.66s**

| Agent | Avg runtime (s) | Critical-path frequency |
|---|---:|---:|
| planner | 0.77 | 100% |
| researcher_b | 3.08 | 100% |
| synthesizer | 0.88 | 100% |
| critic | 0.97 | 100% |
| final_writer | 0.96 | 100% |
| researcher_a | 1.52 | 0% |
| researcher_c | 1.60 | 0% |

Waiting times were negligible (~0.001–0.004s).  
Critical path was always:

```text
planner → researcher_b → synthesizer → critic → final_writer
```

### 5.2 Real LLM Results (5 runs)

Average end-to-end critical-path latency: **~119.3s**

| Agent | Avg runtime (s) | Avg waiting (s) | Critical-path frequency |
|---|---:|---:|---:|
| researcher_b | 41.47 | 0.008 | **80%** |
| synthesizer | 21.49 | 0.001 | 100% |
| planner | 21.40 | 0.000 | 100% |
| final_writer | 20.78 | 0.001 | 100% |
| researcher_c | 19.72 | 0.012 | 20% |
| critic | 13.61 | 0.001 | 100% |
| researcher_a | 10.85 | 0.003 | **0%** |

**Figure 2.** Average agent runtime (LLM).

![Average runtime](llm/avg_runtime.png)

**Figure 3.** Critical-path frequency (LLM).

![Critical-path frequency](llm/critical_path_freq.png)

Most common critical path (4/5 runs):

```text
planner → researcher_b → synthesizer → critic → final_writer
```

One run used `researcher_c` instead of `researcher_b` on the parallel stage.

### 5.3 Comparison

| Aspect | Simulated | Real LLM |
|---|---|---|
| End-to-end latency | ~6.7s | ~119s |
| Parallel bottleneck | researcher_b (100%) | researcher_b (80%) |
| Waiting time | negligible | negligible |
| Runtime variance | low | high (e.g., one B run ~95s) |

---

## 6. Discussion

### 6.1 Answer to the Research Question

**Yes.** Across LLM runs, agent criticality is uneven and fairly stable:

- Among parallel researchers, B dominates end-to-end latency
- Speeding up A would rarely reduce total latency
- Speeding up B would usually reduce total latency
- Sequential agents (planner, synthesizer, critic, writer) are always on the critical path by structure

Waiting time is not the bottleneck in this prototype; **execution time of critical agents** is.

### 6.2 Implications for Scheduling

These results support studying priority-aware scheduling policies such as:

- critical-path-first
- slack-based prioritization
- HEFT-style upward rank

when multiple agent calls compete for limited compute.

### 6.3 Limitations

- Small LLM sample size (5 runs) due to free-tier rate limits
- Single workflow topology
- Single model (`gpt-oss-20b:free`)
- Researcher B’s heavier prompt biases criticality (intentional but should be ablated later)
- No true multi-tenant GPU/API contention yet; waiting times are near zero

---

## 7. Conclusion

This profiling study shows that multi-agent LLM workflows can exhibit clear critical-agent structure. In both simulated and real LLM experiments, one parallel researcher disproportionately determined end-to-end latency, while faster parallel peers rarely appeared on the critical path.

The prototype (LangGraph + tracing + critical-path analysis) is sufficient to motivate the next stage: designing and evaluating critical-agent-aware scheduling. A separate conceptual reproduction of KVFlow (§9) shows that structural STE is useful for **prefix-cache** retention/prefetch but is not a substitute for latency-weighted criticality when deciding **compute** priority.

---

## 8. Next Steps

1. Increase LLM runs for more stable statistics
2. Ablate prompt asymmetry (equal prompts for A/B/C)
3. Implement priority-aware ready-queue scheduling
4. Compare FIFO vs critical-path-first / HEFT-style policies
5. Evaluate under concurrent workflows / limited parallelism
6. Optionally combine critical-path **compute** priority with STE-based **cache** retention in a joint scheduler

---

## 9. KVFlow Conceptual Reproduction (Layers 1–2)

Package [`kvflow/`](../kvflow/) is a pure-Python conceptual reproduction of Pan et al. (KVFlow,
arXiv:2507.07400): Agent Step Graph + workflow-aware prefix-cache policies. It does **not**
run SGLang or real KV tensors.

**Relation to this project:** profiling above motivates scheduling by **compute criticality**
(who runs when under limited parallelism). KVFlow’s STE motivates **cache lifetime** (which
prefixes to keep / prefetch). Same research/writing DAG topology; different control knob.

### 9.1 Layer 1 — Steps-to-execution vs critical path

STE is computed from topology only (planner active at workflow start): active agents get 0;
downstream STE is `agg(pred STEs) + 1` with `max` AND-joins / `min` OR-branches.

Compared against **critical-path frequency** from 20 simulated runs (`simulated/traces.jsonl`):
fraction of runs where each agent lies on the runtime-weighted longest path.

| Agent | Avg runtime (s) | Critical-path freq | STE |
|---|---:|---:|---:|
| planner | 0.77 | 1.0 | 0 |
| researcher_a | 1.52 | 0.0 | 1 |
| researcher_b | 3.08 | 1.0 | 1 |
| researcher_c | 1.60 | 0.0 | 1 |
| synthesizer | 0.88 | 1.0 | 2 |
| critic | 0.97 | 1.0 | 3 |
| final_writer | 0.96 | 1.0 | 4 |

**Figure (Layer 1).** STE vs critical-path frequency:

![STE vs critical-path frequency](../kvflow/results/ste_vs_critical_path.png)

**Finding:** All three researchers share STE = 1 (same structural depth), but only
`researcher_b` is always critical because it is much slower. STE alone cannot replace
latency-weighted critical-path analysis for **compute** prioritization.

### 9.2 Layer 2 — Prefix-cache policy simulation

Synthetic sequential chain of 10 agents (paper-like microbench; `--graph research` also
supported). Tight GPU capacity ≈ shared root + two fixed leaves + one dynamic suffix.
Policies: **LRU**, **STE** eviction (evict highest STE / dynamics first), **STE+prefetch**
(overlap next-agent fixed-prefix load with decode). Hit = fixed leaf already on GPU or
loading. Timings use fixed token/bandwidth constants in `kvflow/cache_sim.py` (reproducible,
not hardware-parity).

| Setting | Policy | Avg latency (s) | Hit rate | Stall (s) | Evictions |
|---|---|---:|---:|---:|---:|
| small | lru | 8.14 | 0.00 | 0.82 | 120 |
| small | ste | 8.12 | 0.10 | 0.74 | 114 |
| small | ste+prefetch | 8.02 | 0.90 | 0.12 | 120 |
| medium | lru | 16.53 | 0.00 | 3.17 | 120 |
| medium | ste | 16.48 | 0.10 | 2.86 | 114 |
| medium | ste+prefetch | 16.07 | 0.90 | 0.40 | 120 |
| large | lru | 34.06 | 0.00 | 12.34 | 120 |
| large | ste | 33.85 | 0.10 | 11.11 | 114 |
| large | ste+prefetch | 32.21 | 0.90 | 1.28 | 120 |

**Figure (Layer 2).** Policy comparison across settings:

![KVFlow cache policy comparison](../kvflow/results/cache_policy_comparison.png)

**Findings:**

- STE slightly beats LRU on hit rate and stall (keeps nearer-future prefixes under pressure).
- STE+prefetch is the clear win: ~90% hit rate and much lower stall, especially at `large`
  fixed prompts, by loading the next agent while the current one decodes.
- Eviction counts stay similar (~114–120); policies mainly change *which* entries survive.

### 9.3 How to reproduce

```bash
# No spaces after commas in --configs
python -m kvflow.compare_criticality --traces simulated/traces.jsonl
python -m kvflow.experiment --configs lru,ste,ste+prefetch --out kvflow/results
python -m unittest discover -s kvflow/tests -v
```

Details: [`kvflow/README.md`](../kvflow/README.md). Artifacts: `kvflow/results/`.

### 9.4 Takeaway for scheduling design

| Knob | Signal | Goal |
|---|---|---|
| This project | Critical-path / slack (latency-weighted) | Prioritize scarce **compute** |
| KVFlow (here) | STE (+ prefetch) | Retain / prefetch **KV prefixes** |

A full system could use both: schedule critical agents first, and keep/prefetch prefixes for
agents that are structurally imminent (low STE).

---

## Appendix A. Repository Layout

```text
critical agent scheduling/
  simulated/     # sleep-based agents, traces, summary
  llm/           # OpenRouter LLM agents, traces, plots
  kvflow/        # KVFlow STE + prefix-cache conceptual sim
  requirements.txt
  env.example
  REPORT.md
```

## Appendix B. How to Reproduce

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
cp env.example .env             # add OPENROUTER_API_KEY

cd simulated
python workflow.py
python analyze.py

cd ../llm
python workflow_llm.py
python analyze_llm.py

# KVFlow Layers 1–2 (conceptual)
cd ..
python -m kvflow.compare_criticality --traces simulated/traces.jsonl
python -m kvflow.experiment --configs lru,ste,ste+prefetch --out kvflow/results
```

## Appendix C. Key References

- Pan et al. 2025. *KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows.* arXiv:2507.07400.
- Pearson et al. 2026. *Graph-Based Agentic AI with LangGraph: Workflow Pathways for Long-Running Stateful Business Processes.* [arXiv:2607.19297](https://arxiv.org/abs/2607.19297).
- H. Topcuoglu, S. Hariri and Min-You Wu, "Performance-effective and low-complexity task scheduling for heterogeneous computing," in IEEE Transactions on Parallel and Distributed Systems, vol. 13, no. 3, pp. 260-274, March 2002, doi: 10.1109/71.993206.
- Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., Zhu, E., ... & Wang, C. (2023). AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. arXiv preprint arXiv:2308.08155.
