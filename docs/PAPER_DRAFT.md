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

**Takeaway:** cache-efficiency wins **generalize** (especially random batch: recompute 1024 vs CPF 31232). Makespan wins stay **small or absent**; CPF sometimes edges Couloir on random batch makespan while losing badly on recompute.

### 5.5 Aging ablation

Single-WF and batch: aging on/off ≈ same mean makespan (`aging_ablation.csv`, batch columns). Kept for starvation-freedom, not as a headline latency claim.

### 5.6 Noise sweep

Duration noise \(\sigma\in\{0,0.1,0.25,0.5\}\) (5 trials). Fan-out ranking Couloir ≤ CPF ≤ FIFO stable in mean; writing near-tied (`noise_sweep.csv`).

### 5.7 Live free-tier (supporting)

Gemini `gemini-flash-lite-latest`, \(K=2\). API-latency dominated; **not** a paper claim.

### 5.8 Limitations / GPU status

Homogeneous workers; conceptual prefill \(P/\lambda\); synthetic chain/fan-out/random durations (writing profiled). **SGLang equal-citizen gate deferred:** sim claim is workshop-credible on the *cache* axis; makespan breadth is still too thin to justify burning the ≤US$50 GPU budget yet. Next sim steps before GPU: harder contention (tighter \(C\), higher \(M/K\)), weight grid under batch only.

---

## 6. Discussion

Couloir’s durable contribution is **joint reuse under multi-tenant arrivals**, not a universal makespan hammer. Single-WF fan-out (~2% over CPF) does not generalize; batch hit-rate/recompute does. Weight defaults are provisional—reuse is necessary under batch; other terms need a regime where dispatch order is contested.

---

## 7. Conclusion

Couloir unifies compute priority and prefix-cache control. Discrete-event evidence supports a **cache-joint** claim under batch/random multi-arrival, with a narrow single-WF fan-out makespan signal. Softcopy + regenerable artifacts target arXiv cs.DC and an MLSys/serving workshop ([`VENUES.md`](VENUES.md)); real-engine validation remains optional pending stronger sim breadth.

---

## References

1. H. Topcuoglu, S. Hariri, Min-You Wu. Performance-effective and low-complexity task scheduling for heterogeneous computing. *IEEE TPDS*, 2002.
2. Pan et al. KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows. arXiv:2507.07400, 2025.
3. Wu et al. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation. arXiv:2308.08155, 2023.
4. Pearson et al. Graph-Based Agentic AI with LangGraph. arXiv:2607.19297, 2026.

---

## Appendix A — Pitch (deferred)

> We have a working discrete-event sim + baseline matrix where Couloir jointly beats CPF→STE on cache reuse and improves fanout makespan; looking for a mentor/co-author to sharpen the claim vs KVFlow/TOPAS and target a workshop after arXiv.
