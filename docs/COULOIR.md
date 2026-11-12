# Couloir

Joint **compute + prefix-cache** scheduling for multi-agent LLM DAGs under shared worker slots `K` and cache capacity `C`.

## Claim

Under contention (`K` smaller than ready parallelism and/or tight `C`), a single utility that trades off critical-path progress, slack, cache reuse, eviction cost, and aging beats optimizing **either** compute priority **or** cache lifetime alone.

## Glossary

| Term | Meaning |
| --- | --- |
| **K** | Number of homogeneous worker slots (API / GPU decode slots in the sim). |
| **C** | Prefix-cache capacity in tokens (conceptual GPU KV budget). |
| **CPF** | Critical-Path-First: schedule highest remaining upward rank first. |
| **STE** | Steps-to-execution (KVFlow): temporal proximity used for cache eviction. |
| **Couloir** | Joint policy: one score drives ready-queue order **and** keep/evict/prefetch. |
| **Sim track** | Discrete-event simulator in `scheduler/` — source of truth for the claim. |
| **Engine track** | Optional SGLang + local 8B case study in `engine/` (equal-citizen after gate). |
| **Equal-citizen gate** | Sim ablations done **and** ≥30 workflow runs on DAG₁ + ≥20 on DAG₂ with matching directionality. |
| **Free-tier live** | Live LLM validation must use $0 API tiers; paid budget is GPU-only (≤ US$50). |

## Utility

For ready task \(i\) at time \(t\):

\[
U(i,t)=w_{\mathrm{cp}}\,\hat{r}_{\mathrm{cp}}(i)+w_{\mathrm{slack}}(1-\hat{r}_{\mathrm{slack}}(i))+w_{\mathrm{reuse}}\,\mathrm{reuse}(i)-w_{\mathrm{evict}}\,\mathrm{evict}(i)+w_{\mathrm{age}}\,\hat{a}(i,t)
\]

- \(r_{\mathrm{cp}}\): upward rank / remaining critical path (expected durations).
- \(r_{\mathrm{slack}}\): deadline − (now + remaining CP); lower slack ⇒ higher urgency.
- \(\mathrm{reuse}\): 1 if agent prefix already on GPU, else 0.
- \(\mathrm{evict}\): capacity pressure to admit the agent’s prefix ∈ [0, 1].
- \(a(i,t)\): wait since `ready_time`; hats denote min–max normalize over the current ready set.

Default weights: `cp=1`, `slack=1`, `reuse=0.75`, `evict=0.5`, `age=1.25` (`scheduler/couloir.py`).

Aging is required: without it, short non-critical tasks can starve under sustained critical arrivals.

## Cache actions

When Couloir selects task \(i\):

1. **Keep** \(i\) and up to `prefetch_top_k` other high-utility ready agents.
2. **Evict** on-GPU agents not in the keep set (lowest utility / highest STE first).
3. **Prefetch** keep-set agents other than \(i\) (conceptual; used by joint engine).

Prefetch hits shorten task duration by \(P/\lambda_{\mathrm{prefill}}\) (`PREFILL_TOKENS_PER_SEC` in `scheduler/engine.py`) so cache policy can move makespan, not only hit-rate counters.

## Baselines

- Compute-only: FIFO, CPF, LSF, Simplified HEFT.
- Cache-only: LRU, STE, STE+prefetch (`kvflow/`).
- Ablation: two-phase CPF then STE (sequential knobs) vs joint Couloir.

## Code

| Module | Role |
| --- | --- |
| `scheduler/couloir.py` | Pure scoring + cache action planning |
| `scheduler/policies.py` | Registers `couloir` policy |
| `scheduler/engine.py` | Discrete-event workers; optional joint cache |
| `kvflow/` | Conceptual prefix-cache simulator |

## Paper path (not FYP)

Finish sim claim → draft → pitch professors for mentor/co-author → arXiv → workshop. Scholarships / FYP milestones are out of scope.
