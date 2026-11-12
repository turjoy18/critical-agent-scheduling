# Couloir — Joint Compute + Prefix-Cache Scheduling

**Couloir** schedules multi-agent LLM DAGs under shared worker slots `K` and prefix-cache capacity `C` with **one utility** for ready-queue order and keep/evict/prefetch.

| Doc | Purpose |
|-----|---------|
| [`docs/PAPER_DRAFT.md`](docs/PAPER_DRAFT.md) | Softcopy paper (sim-primary) |
| [`docs/COULOIR.md`](docs/COULOIR.md) | Claim + glossary |
| [`docs/STATUS.md`](docs/STATUS.md) | **Freeze / pause posture** |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Regen commands |
| [`docs/VENUES.md`](docs/VENUES.md) | arXiv / workshop (mentor-shaped) |

> Two knobs, one policy: **who runs when** and **which KV prefixes to keep**.

**Sim track is frozen.** No new weight grids or DAG shapes unless a mentor asks. Mentor pitch and GPU are paused (months); see [`docs/STATUS.md`](docs/STATUS.md).

## Tracks

| Track | Role | Status |
|-------|------|--------|
| **Sim** | Source of truth | **Frozen** — matrix, scoped-cache batch, softcopy |
| **Live free-tier** | Gemini smoke | Done — not a paper claim |
| **Engine** | SGLang case study | **Paused** — Colab first, only after mentor |

**Cost policy:** free LLM APIs only; GPU ≤ US$50 if ever resumed.

## Quick start (reproduce frozen artifacts)

```bash
python -m venv .venv
# Windows Git Bash: source .venv/Scripts/activate
# Linux/macOS:      source .venv/bin/activate
pip install -r requirements.txt

python -m unittest discover -s scheduler/tests -v
# Optional full regen (do not treat as “keep iterating”):
# see docs/REPRODUCIBILITY.md
```

Primary claim numbers: batch \(M=20\), **workflow-scoped** prefixes — see softcopy §5.5.

| Shape | Couloir recompute | CPF recompute | Couloir makespan | CPF makespan |
|-------|-------------------|---------------|------------------|--------------|
| fanout | **2048** | 75776 | **587** | 604 |
| random | **1536** | 76800 | **698** | 717 |

## Layout

| Path | Purpose |
|------|---------|
| `scheduler/` | Discrete-event Couloir + frozen sweep CSVs |
| `kvflow/` | Conceptual STE/LRU sim |
| `llm/`, `simulated/` | Profiling (historical) |
| `docs/` | Paper + status |

## References

- Pan et al. 2025. *KVFlow.* arXiv:2507.07400.
- Topcuoglu et al. 2002. HEFT. *IEEE TPDS*.
- Wu et al. 2023. *AutoGen.* arXiv:2308.08155.
