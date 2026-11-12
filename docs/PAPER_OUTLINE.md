# Couloir — paper outline

**Working title:** Couloir: Joint Compute and Prefix-Cache Scheduling for Multi-Agent LLM Workflows

**Claim:** Under shared worker slots \(K\) and prefix-cache capacity \(C\), a single utility over remaining critical path, slack, reuse benefit, eviction cost, and aging reduces makespan / JCT vs compute-only or cache-only (and vs two-phase CPF→STE).

## 1. Introduction
- Multi-agent DAGs + serving contention
- Two knobs treated separately in prior work → joint decision
- Contributions: Couloir utility; discrete-event eval; multi-DAG matrix; optional SGLang case study

## 2. Related work
- HEFT / critical-path scheduling
- KVFlow (STE + prefetch), Continuum (TTL), TOPAS (joint prefix-state), CacheWise
- Multi-agent frameworks (AutoGen, LangGraph) — application layer, not serving policy

## 3. Model
- Workflow DAG, homogeneous workers \(K\), token cache capacity \(C\)
- Metrics: makespan, JCT p50/p99, hit rate, recompute tokens

## 4. Couloir
- Utility formula (see `docs/COULOIR.md`)
- Cache keep / evict / prefetch from same ranking
- Aging / starvation argument

## 5. Evaluation (sim)
- DAGs: writing, chain, fanout
- Baselines: FIFO, CPF, HEFT, LRU, STE, two-phase CPF→STE, Couloir
- Sweeps: \(K\), \(C\), noise (artifacts under `scheduler/results/couloir_sweeps/`)
- Ablation: disable aging; two-phase vs joint

## 6. Live free-tier validation (supporting)
- CPF / Couloir vs FIFO at \(K=2\) on one free backend
- Not primary claim; external validity caveat (API variance)

## 7. Optional engine case study
- SGLang + 8B; equal-citizen gate (30+20 runs) if GPU available

## 8. Discussion / limitations
- Homogeneous workers; mean-duration model; conceptual cache tokens
- Comparison scope vs full serving stacks

## 9. Conclusion

## Professor pitch (1 sentence)
> We have a working sim + baseline matrix for Couloir; looking for a mentor/co-author to sharpen the claim vs KVFlow/TOPAS and target a workshop after arXiv.
