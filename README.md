# Couloir — Joint Compute + Prefix-Cache Scheduling

**Couloir** schedules multi-agent LLM DAGs under shared worker slots `K` and prefix-cache capacity `C` with **one utility** for ready-queue order and keep/evict/prefetch.

Paper softcopy: [`docs/PAPER_DRAFT.md`](docs/PAPER_DRAFT.md) · Charter: [`docs/COULOIR.md`](docs/COULOIR.md) · Regen: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)

> Two knobs, one policy: **who runs when** and **which KV prefixes to keep**.

## Tracks

| Track | Role | Status |
|-------|------|--------|
| **Sim** (`scheduler/`) | Source of truth for the claim | Done — matrix, ablations, figures |
| **Live free-tier** | Gemini smoke (not paper claims) | Done — [`docs/LIVE_BACKEND.md`](docs/LIVE_BACKEND.md) |
| **Engine** (optional) | SGLang + 8B case study | Deferred — [`docs/GPU_CLOUD.md`](docs/GPU_CLOUD.md), ≤US$50 |

**Cost policy:** free LLM APIs only; paid budget is GPU-only (≤ US$50).

## Quick start (<15 min)

```bash
python -m venv .venv
# Windows Git Bash: source .venv/Scripts/activate
# Linux/macOS:      source .venv/bin/activate
pip install -r requirements.txt

python -m unittest discover -s scheduler/tests -v
python -m scheduler.couloir_sweep --all-dags --with-ablations
python -m scheduler.weight_ablation
python -m scheduler.batch_couloir
python -m scheduler.plot_couloir
```

Figures land in `scheduler/results/couloir_sweeps/figures/`. Weight + batch/random CSVs land beside them.

### Optional live smoke

```bash
cp env.example .env   # set GEMINI_API_KEY
python -m scheduler.live_runner --backend gemini --runs 2 --workers 2
```

### Legacy profiling / KVFlow conceptual sims

```bash
cd simulated && python workflow.py && python analyze.py
python -m kvflow.experiment --configs lru,ste,ste+prefetch --out kvflow/results
python -m scheduler.experiment --out scheduler/results
```

## Layout

| Path | Purpose |
|------|---------|
| `scheduler/` | Discrete-event engine, Couloir, sweeps, live_runner |
| `kvflow/` | Conceptual STE/LRU prefix-cache sim |
| `llm/`, `simulated/` | Profiling agents + critical-path analysis |
| `docs/PAPER_DRAFT.md` | Softcopy paper |
| `docs/VENUES.md` | arXiv / workshop plan |

## Headline sim results (batch \(M=20\), workflow-scoped prefixes)

| Shape | Couloir recompute | CPF recompute | Couloir makespan | CPF makespan |
|-------|-------------------|---------------|------------------|--------------|
| fanout | **2048** | 75776 | **587** | 604 |
| random | **1536** | 76800 | **698** | 717 |

Primary claim = **cache-joint under multi-arrival**, not single-WF makespan. See `docs/PAPER_DRAFT.md`.

## Status

- [x] Couloir utility + joint cache actions
- [x] Multi-DAG sweeps (writing / chain / fanout) + aging/noise ablations
- [x] Paper figures + softcopy draft
- [x] Free-tier Gemini live smoke
- [ ] Optional SGLang equal-citizen gate
- [ ] arXiv upload (author list pending)

## References

- Pan et al. 2025. *KVFlow.* arXiv:2507.07400.
- Topcuoglu et al. 2002. HEFT. *IEEE TPDS*.
- Wu et al. 2023. *AutoGen.* arXiv:2308.08155.
