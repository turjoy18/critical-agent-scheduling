# Couloir: Joint Compute and Prefix-Cache Scheduling for Multi-Agent LLM Workflows

**Status:** softcopy draft (sim-primary). Figures: `scheduler/results/couloir_sweeps/figures/`. Regen: see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## Abstract

Multi-agent LLM workflows expose two scarce resources under shared serving: worker slots \(K\) and prefix-cache capacity \(C\). Prior systems optimize ready-queue priority or cache residency in isolation. We present **Couloir**, a single utility that jointly ranks ready tasks and keep/evict/prefetch decisions. In a discrete-event simulator with **workflow-scoped** prefixes (no cross-workflow type freeloading), Couloir’s durable result is a **cache-joint** win under multi-arrival batches: large recompute cuts vs CPF+STE / FIFO+LRU, plus a consistent ~2–3% makespan edge at \(M\in\{5,10,20\}\). That edge is mostly “CPF collapses without freeloading,” not Couloir speeding up—scoped Couloir makespan barely moves vs type-keyed mode. Hits under scoping are largely **prefetch** (each agent runs once per workflow). Aging is a starvation guard. Live Gemini is smoke only. Optional SGLang validation is unlocked, not required.

---

## 1. Introduction

Agent frameworks (AutoGen, LangGraph) compose planner → parallel researchers → join → critic → writer DAGs. End-to-end latency is dominated by (i) **who runs when** under limited decode/API slots and (ii) **which KV prefixes stay resident** when GPU memory is tight. HEFT-style critical-path scheduling addresses (i); KVFlow STE/prefetch and related cache policies address (ii). Production stacks (vLLM, SGLang) expose reuse APIs but leave joint DAG-aware control to the application.

**Claim.** Under shared \(K\) and \(C\), one utility over critical-path progress, slack, reuse, eviction pressure, and aging **improves prefix residency (via joint prefetch) and reduces recompute** versus compute-only, cache-only, or two-phase CPF→STE—especially under multi-arrival batches with workflow-scoped keys. Makespan gains are secondary (~2–3% batch under scoping; driven largely by baseline collapse without cross-WF freeloading).

**Contributions.**

1. Couloir joint utility and cache action planner (`scheduler/couloir.py`).
2. Discrete-event eval: multi-DAG matrix, batch/random, weight grid, **workflow-scoped vs type-keyed** cache sensitivity.
3. Free-tier live smoke (Gemini); cost policy: free chat APIs; GPU ≤US$50 only.
4. Softcopy + regenerable artifacts toward arXiv / workshop.

---

## 2. Related work

**DAG / critical-path scheduling.** HEFT ranks by upward rank under heterogeneous processors [Topcuoglu et al., TPDS 2002]. We use homogeneous CPF / simplified HEFT as compute-only baselines.

**Prefix-cache for agent workflows.** KVFlow introduces STE and STE+prefetch [Pan et al., arXiv:2507.07400]. Continuum-style TTL and TOPAS-style joint prefix-state optimize residency with ready-queue order treated as external. Serving engines provide RadixAttention / prefix APIs without DAG joint utilities.

**Multi-agent frameworks.** AutoGen and LangGraph orchestrate graphs; they do not schedule GPU slots or KV eviction under shared \(K,C\).

**Gap.** Couloir jointly decides dispatch and residency; two-phase CPF→STE is the sequential-knob ablation.

---

## 3. Model

A workflow is a DAG of agent tasks with expected durations from LLM profiling means (writing) or synthetic means (chain, fan-out, random layered). \(K\) homogeneous workers. Prefix-cache capacity \(C\) holds prefixes of size \(P\). Profiled duration assumes cold prefill; a hit subtracts \(P/\lambda_{\mathrm{prefill}}\) (\(\lambda=1024\) tok/s). Metrics: makespan, JCT p50/p99, hit rate, recompute tokens.

**Prefix keying (important).** Default `cache_key_mode=workflow_agent` keys residency by `workflow_id:agent_type`, so workflows do **not** share prefixes merely by sharing an agent-type string. The legacy `agent_type` mode enables cross-workflow reuse and **inflates** batch hit rates; sensitivity only. Do not quote type-keyed hit rates as KV hits.

**Prefetch, not cross-repeat reuse.** Under scoped keys each agent type runs once per workflow, so a “hit” is almost always Couloir **prefetching** the keep-set (`prefetch_top_k`) onto the GPU before dispatch—not reuse of a prior run of the same agent. Scoped hit rates ~0.96–0.98 vs CPF ~0.06–0.10 are therefore a **prefetch gap** from the joint utility, which is still the claim (dispatch order deciding what to prefetch).

**DAGs:** writing; chain; fan-out; random layered (breadth under batch; solo random is null vs CPF).

---

## 4. Couloir

For ready task \(i\) at time \(t\):

\[
U(i,t)=w_{\mathrm{cp}}\,\hat{r}_{\mathrm{cp}}(i)+w_{\mathrm{slack}}(1-\hat{r}_{\mathrm{slack}}(i))+w_{\mathrm{reuse}}\,\mathrm{reuse}(i)-w_{\mathrm{evict}}\,\mathrm{evict}(i)+w_{\mathrm{age}}\,\hat{a}(i,t)
\]

Defaults \(w=(1,0.25,1.5,0.5,0.25)\) (consensus after batch grid; shape-best often uses `reuse=0.75` with near-tied metrics—not a sharp optimum). Hats = min–max over the ready set. Same ranking drives keep / evict / prefetch. **Aging** = starvation guard; not a mean-makespan claim.

**Baselines:** FIFO, CPF, HEFT; FIFO+LRU; CPF+STE; two-phase CPF→STE; Couloir.

---

## 5. Evaluation

**Setup.** \(K\in\{1,2,3\}\), \(P=512\), \(C\in\{1024,2048,4096\}\). Artifact: `couloir_baseline_matrix.csv`. Commands in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

### 5.1 Makespan (\(C=2048\))

| DAG | \(K\) | FIFO | CPF / HEFT | Couloir |
|-----|-------|------|------------|---------|
| writing | 2 | 118.75 | 118.75 | 118.75 |
| writing | 1 | 149.32 | 149.32 | **148.32** |
| chain | 2 | 89.00 | 89.00 | 89.00 |
| fanout | 2 | 72.00 | 68.00 | **66.50** |
| fanout | 3 | 56.00 | 48.00 | **47.50** |
| fanout | 1 | 120.00 | 120.00 | **117.50** |

Fan-out: CPF already beats FIFO by ranking long branches; Couloir’s prefetch hits further cut prefill on the critical path. Writing/chain: single-shot agent types put many hit savings off the critical path → makespan often ties CPF.

### 5.2 Cache efficiency (\(K=2\), \(C=2048\))

| DAG | Policy | Hit rate | Recompute tokens |
|-----|--------|----------|------------------|
| writing | FIFO+LRU | 0.00 | 3584 |
| writing | CPF+STE / 2P+STE | 0.14 | 3072 |
| writing | **Couloir** | **0.29** | **2560** |
| fanout | FIFO+LRU | 0.00 | 4096 |
| fanout | CPF+STE / 2P+STE | 0.12 | 3584 |
| fanout | **Couloir** | **0.62** | **1536** |
| chain | all | 0.00 | 3584 |

Two-phase matches CPF+STE on cache metrics, not Couloir—supporting the **joint** claim on the reuse axis.

### 5.3 Weight sensitivity and leave-one-out

**Single workflow:** dropping any term leaves fan-out makespan at 66.5 s and writing at 118.75 s (`weight_leave_one_out.csv`, `weight_sensitivity.csv`). Ready-set structure is too constrained for weights to reshuffle order.

**Batch \(M=10\), \(K=4\), stagger 2 s** (`weight_batch_leave_one_out.csv`):

| Shape | Config | Makespan | Hit rate | Recompute |
|-------|--------|----------|----------|-----------|
| fanout | default | 295.5 | 0.91 | 3584 |
| fanout | drop **reuse** | **300.0** | **0.56** | **17920** |
| fanout | drop age | 295.0 | 0.98 | 1024 |
| random | default | 333.8 | 0.98 | 1024 |
| random | drop **reuse** | 335.3 | 0.88 | 5120 |

**Reuse is the load-bearing cache term under multi-arrival.** CP/slack/evict/age barely move batch makespan on these shapes; aging remains a starvation guard (unit-tested) rather than a mean-makespan lever.

### 5.4 Random DAGs + batch multi-arrival

24 solo random layered DAGs (\(K=2\)): Couloir/CPF/two-phase **tie** mean makespan (82.2 s); Couloir/CPF hit rate 0.27 vs FIFO 0.00 (`random_solo_summary.csv`).

Batch \(M\in\{1,5,10,20\}\), \(K=4\), \(C=2048\), stagger 2 s (`batch_multi_arrival.csv`):

| Shape | \(M\) | FIFO makespan | CPF makespan | Couloir makespan | Couloir hit | CPF hit |
|-------|-------|---------------|--------------|------------------|-------------|---------|
| fanout | 20 | 600.5 | 587.5 | **586.5** | **0.95** | 0.92 |
| random | 20 | 727.6 | **706.2** | 709.1 | **0.99** | 0.62 |
| writing | 20 | 746.9 | 735.1 | **734.5** | **0.99** | 0.98 |

**Takeaway (type-keyed, pre-fix):** relative recompute gaps look huge but are inflated by cross-WF type sharing. Makespan batch “wins” at M=20 are **noise** (≤1 s); CPF often ahead on random/fanout M=5–10. Solo random DAGs are a **null** test (Couloir≡CPF).

### 5.5 Cache keying sensitivity (primary batch claim)

`cache_scope_sensitivity.csv` — \(M=20\), \(K=4\), \(C=2048\):

| Shape | Mode | Couloir makespan | CPF makespan | Couloir hit | CPF hit | Couloir recompute | CPF recompute |
|-------|------|------------------|--------------|-------------|---------|-------------------|---------------|
| fanout | workflow_agent | **587** | 604 | 0.98 | 0.08 | **2048** | 75776 |
| random | workflow_agent | **698** | 717 | 0.98 | 0.06 | **1536** | 76800 |
| writing | workflow_agent | **736** | 751 | 0.96 | 0.10 | **2560** | 64512 |

**Scoped recompute gap survives.** Couloir absolute makespan barely moves vs type-keyed (e.g. fan-out 587 vs 586.5); the ~2–3% batch edge is **CPF collapsing** without freeloading (604 / 717 / 751). Type-keyed remains an optimistic upper bound. Mechanism: scoped hits ≈ prefetch (see §3).

### 5.6 Batch weight grid

`batch_weight_grid.csv` (324 rows). Consensus default \(w=(1,0.25,1.5,0.5,0.25)\); grid often ties `reuse=0.75` on hit/recompute (±1 s makespan)—do not oversell as the unique optimum.

### 5.7 Aging ablation

Near-zero mean-makespan effect; drop-age can even improve batch fan-out hit/recompute. Kept for starvation-freedom only.

### 5.8 Noise + live

Noise sweep: fan-out ranking stable in mean. Live Gemini: smoke only.

### 5.9 Limitations / GPU status

Conceptual prefill; synthetic non-writing DAGs. **Scoped cache gap survived → GPU stage is unlocked** for optional equal-citizen evidence (≤US$50, [`GPU_CLOUD.md`](GPU_CLOUD.md)). Still not required for arXiv/workshop softcopy.

---

## 6. Discussion

Lead with **joint prefetch under multi-arrival with workflow-scoped keys**. Makespan edge under scoping ≈ CPF losing freeloaded hits. Type-keyed numbers are an optimistic upper bound, not a KV claim.

---

## 7. Conclusion

Couloir unifies compute priority and prefix-cache control. With workflow-scoped prefixes, batch multi-arrival shows large recompute cuts (and modest makespan edges) vs CPF+STE. Softcopy targets arXiv cs.DC + serving workshop; optional SGLang validation is now justified, not mandatory ([`VENUES.md`](VENUES.md)).

---

---

## References

1. H. Topcuoglu, S. Hariri, Min-You Wu. Performance-effective and low-complexity task scheduling for heterogeneous computing. *IEEE TPDS*, 2002.
2. Pan et al. KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows. arXiv:2507.07400, 2025.
3. Wu et al. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. arXiv:2308.08155, 2023.
4. Pearson et al. Graph-Based Agentic AI with LangGraph. arXiv:2607.19297, 2026.

---

## Appendix A — Pitch (deferred)

> We have a Couloir sim where joint dispatch+prefetch beats CPF→STE on **recompute under batch multi-arrival** (workflow-scoped prefixes); makespan is ~tied. Looking for a systems/LLM-serving mentor to harden the claim vs KVFlow/TOPAS toward arXiv + workshop.
