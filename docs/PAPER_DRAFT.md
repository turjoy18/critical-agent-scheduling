# Couloir: Joint Compute and Prefix-Cache Scheduling for Multi-Agent LLM Workflows

**Status:** softcopy draft (sim-primary). Figures: `scheduler/results/couloir_sweeps/figures/`. Regen: see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## Abstract

Multi-agent LLM workflows expose two scarce resources under shared serving: worker slots \(K\) and prefix-cache capacity \(C\). Prior systems optimize ready-queue priority or cache residency in isolation. We present **Couloir**, a single utility that ranks ready agents by remaining critical path, slack, reuse benefit, eviction cost, and aging, and uses the same ranking for keep / evict / prefetch. In a discrete-event simulator across writing, chain, and fan-out DAGs, Couloir matches or beats compute-only and two-phase CPF→STE baselines: on fan-out at \(K=2\), makespan drops from 72 s (FIFO) and 68 s (CPF) to **66.5 s**, with cache hit rate **0.62** vs **0.12** for STE. On writing/chain, makespan ties CPF while hit rate and recompute tokens improve. Live free-tier Gemini runs validate integration only. An optional SGLang engine track remains future work under a ≤US$50 GPU budget.

---

## 1. Introduction

Agent frameworks (AutoGen, LangGraph) compose planner → parallel researchers → join → critic → writer DAGs. End-to-end latency is dominated by (i) **who runs when** under limited decode/API slots and (ii) **which KV prefixes stay resident** when GPU memory is tight. HEFT-style critical-path scheduling addresses (i); KVFlow STE/prefetch and related cache policies address (ii). Production stacks (vLLM, SGLang) expose reuse APIs but leave joint DAG-aware control to the application.

**Claim.** Under shared \(K\) and \(C\), one utility over critical-path progress, slack, reuse, eviction pressure, and aging reduces makespan and recompute versus optimizing either knob alone—and versus sequencing CPF then STE.

**Contributions.**

1. Couloir joint utility and cache action planner (`scheduler/couloir.py`).
2. Discrete-event evaluation with multi-DAG matrix, aging ablation, and noise sweeps.
3. Free-tier live smoke path (Gemini) documenting cost policy: free chat APIs; paid budget GPU-only.
4. Softcopy draft + reproducibility harness toward arXiv / workshop.

---

## 2. Related work

**DAG / critical-path scheduling.** HEFT ranks by upward rank under heterogeneous processors [Topcuoglu et al., TPDS 2002]. We use homogeneous CPF / simplified HEFT as compute-only baselines.

**Prefix-cache for agent workflows.** KVFlow introduces STE and STE+prefetch [Pan et al., arXiv:2507.07400]. Continuum-style TTL and TOPAS-style joint prefix-state optimize residency with ready-queue order treated as external. Serving engines provide RadixAttention / prefix APIs without DAG joint utilities.

**Multi-agent frameworks.** AutoGen and LangGraph orchestrate graphs; they do not schedule GPU slots or KV eviction under shared \(K,C\).

**Gap.** Couloir jointly decides dispatch and residency; two-phase CPF→STE is the sequential-knob ablation.

---

## 3. Model

A workflow is a DAG of agent tasks with expected durations from LLM profiling means (writing) or synthetic means (chain, fan-out). \(K\) homogeneous workers. Prefix-cache capacity \(C\) holds agent-keyed prefixes of size \(P\). Profiled duration assumes cold prefill; a hit subtracts \(P/\lambda_{\mathrm{prefill}}\) with \(\lambda_{\mathrm{prefill}}=1024\) tok/s. Metrics: makespan, JCT p50/p99, hit rate, recompute tokens.

**DAGs:** writing (7 agents, 3-way research parallel); chain (linear); fan-out (wide parallel then join).

---

## 4. Couloir

For ready task \(i\) at time \(t\):

\[
U(i,t)=w_{\mathrm{cp}}\,\hat{r}_{\mathrm{cp}}(i)+w_{\mathrm{slack}}(1-\hat{r}_{\mathrm{slack}}(i))+w_{\mathrm{reuse}}\,\mathrm{reuse}(i)-w_{\mathrm{evict}}\,\mathrm{evict}(i)+w_{\mathrm{age}}\,\hat{a}(i,t)
\]

Defaults \(w=(1,1,0.75,0.5,1.25)\). Hats = min–max over the ready set. Top score is dispatched; the same ranking drives keep / evict / prefetch. Aging prevents starvation of short non-critical tasks under sustained critical arrivals.

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

### 5.3 Aging ablation

On fan-out / writing at \(K=2\), Couloir with aging vs without (see `aging_ablation.csv`). Aging is required for starvation-freedom under adversarial ready sets (unit-tested); effect sizes on these single-workflow means are small—batch / multi-arrival settings are the intended stress case (future).

### 5.4 Noise sweep

Duration noise \(\sigma\in\{0,0.1,0.25,0.5\}\) (5 trials, seed 42). Mean makespan rankings are stable: Couloir ≤ CPF+STE ≤ FIFO+LRU on fan-out; writing remains near-tied on makespan with Couloir best on recompute (`noise_sweep.csv`).

### 5.5 Live free-tier (supporting)

Gemini `gemini-flash-lite-latest`, \(K=2\), 2 tasks × {CPF, FIFO, Couloir}. Example makespans (s): task1 32.7 / 24.0 / 9.9; task2 12.1 / 32.6 / 27.8. Dominated by API latency; **not** a paper claim.

### 5.6 Limitations

Homogeneous workers; mean-duration model; conceptual token cache; no SGLang equal-citizen gate yet ([`GPU_CLOUD.md`](GPU_CLOUD.md)). Prefill coupling \(P/\lambda\) is an explicit modeling choice.

---

## 6. Discussion

Couloir’s wins concentrate where (a) parallel width exceeds \(K\) and (b) prefetch can place prefixes on the critical path before dispatch. Linear DAGs and off-path hits improve efficiency metrics without moving makespan—honest scope for the claim. Engine validation would test whether the same utility maps onto real KV events.

---

## 7. Conclusion

Couloir unifies compute priority and prefix-cache control for multi-agent LLM DAGs. Discrete-event results show joint gains on fan-out makespan and consistent cache-efficiency wins over STE and two-phase CPF→STE. We release the simulator, sweeps, figures, and this softcopy toward arXiv and a systems/MLSys workshop ([`VENUES.md`](VENUES.md)).

---

## References

1. H. Topcuoglu, S. Hariri, Min-You Wu. Performance-effective and low-complexity task scheduling for heterogeneous computing. *IEEE TPDS*, 2002.
2. Pan et al. KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows. arXiv:2507.07400, 2025.
3. Wu et al. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. arXiv:2308.08155, 2023.
4. Pearson et al. Graph-Based Agentic AI with LangGraph. arXiv:2607.19297, 2026.

---

## Appendix A — Pitch (deferred)

> We have a working discrete-event sim + baseline matrix where Couloir jointly beats CPF→STE on cache reuse and improves fanout makespan; looking for a mentor/co-author to sharpen the claim vs KVFlow/TOPAS and target a workshop after arXiv.
