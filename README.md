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
python -m scheduler.plot_couloir
```

Figures land in `scheduler/results/couloir_sweeps/figures/`.

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

## Headline sim results (fan-out, \(K=2\), \(C=2048\))

| Policy | Makespan | Hit rate |
|--------|----------|----------|
| FIFO (+LRU) | 72.0 | 0.00 |
| CPF (+STE) | 68.0 | 0.12 |
| **Couloir** | **66.5** | **0.62** |

Writing/chain: makespan often ties CPF; Couloir still cuts recompute tokens. Details in the draft.

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
