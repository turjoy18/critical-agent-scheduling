# Couloir

Joint **compute + prefix-cache** scheduling for multi-agent LLM DAGs under shared worker slots `K` and cache capacity `C`.

## Claim

Under multi-tenant contention (batch arrivals, tight `C`), a single utility that jointly ranks ready tasks and keep/evict/prefetch decisions **improves prefix reuse / cuts recompute** versus compute-only or cache-only (and versus two-phase CPF→STE). Makespan gains are **narrow and workload-dependent** (sometimes ~tied with CPF); do not treat Couloir as a universal latency hammer.

## Glossary

| Term | Meaning |
| --- | --- |
| **K** | Number of homogeneous worker slots (API / GPU decode slots in the sim). |
| **C** | Prefix-cache capacity in tokens (conceptual GPU KV budget). |
| **CPF** | Critical-Path-First: schedule highest remaining upward rank first. |
| **STE** | Steps-to-execution (KVFlow): temporal proximity used for cache eviction. |
| **Couloir** | Joint policy: one score drives ready-queue order **and** keep/evict/prefetch. |
| **cache_key_mode** | `workflow_agent` (default): no cross-workflow prefix sharing. `agent_type`: optimistic shared types (sensitivity only). |
| **Sim track** | Discrete-event simulator in `scheduler/` — source of truth for the claim. |
| **Engine track** | Optional SGLang + local 8B case study (equal-citizen after gate); deferred while sim breadth is thin. |
| **Free-tier live** | Live LLM validation must use $0 API tiers; paid budget is GPU-only (≤ US$50). |

## Utility

For ready task \(i\) at time \(t\):

\[
U(i,t)=w_{\mathrm{cp}}\,\hat{r}_{\mathrm{cp}}(i)+w_{\mathrm{slack}}(1-\hat{r}_{\mathrm{slack}}(i))+w_{\mathrm{reuse}}\,\mathrm{reuse}(i)-w_{\mathrm{evict}}\,\mathrm{evict}(i)+w_{\mathrm{age}}\,\hat{a}(i,t)
\]

Default weights (batch-retuned): `cp=1`, `slack=0.25`, `reuse=1.5`, `evict=0.5`, `age=0.25` (`scheduler/couloir.py`).

**Aging** is a starvation guard (unit-tested under adversarial ready sets). It is **not** a mean-makespan claim; batch ablations show near-zero or slightly worse latency when age is large.

## Cache actions

When Couloir selects task \(i\):

1. **Keep** \(i\) and up to `prefetch_top_k` other high-utility ready prefixes.
2. **Evict** on-GPU prefixes not in the keep set.
3. **Prefetch** keep-set prefixes other than \(i\).

Prefetch hits shorten duration by \(P/\lambda_{\mathrm{prefill}}\) (conceptual). Default prefix key is `workflow_id:agent_type`.

## Baselines

- Compute-only: FIFO, CPF, LSF, Simplified HEFT.
- Cache-only: LRU, STE (`kvflow/` + engine modes).
- Ablation: two-phase CPF then STE vs joint Couloir.

## Code

| Module | Role |
| --- | --- |
| `scheduler/couloir.py` | Scoring + cache action planning |
| `scheduler/engine.py` | Discrete-event workers; joint cache; key modes |
| `scheduler/cache_scope_sensitivity.py` | type-keyed vs workflow-scoped |
| `scheduler/batch_weight_grid.py` | Batch weight grid |

## Paper path

Sim cache-joint claim → softcopy → arXiv / workshop. Pitch systems/LLM-serving mentors (not quantum). Scoped-cache recompute gap survived → optional GPU ≤US$50 equal-citizen track is unlocked.
