# Reproducibility

## One-command paper artifacts

```bash
python -m unittest discover -s scheduler/tests -v
python -m scheduler.couloir_sweep --all-dags --with-ablations
python -m scheduler.weight_ablation
python -m scheduler.batch_couloir
python -m scheduler.cache_scope_sensitivity
python -m scheduler.batch_weight_grid
python -m scheduler.plot_couloir
```

Default engine cache keying: **`workflow_agent`** (no cross-WF type sharing).

Outputs under `scheduler/results/couloir_sweeps/`:

| File | Contents |
|------|----------|
| `couloir_baseline_matrix.csv` | \(K\times C\) × policies × DAGs |
| `cache_scope_sensitivity.csv` | type-keyed vs workflow-scoped batch |
| `batch_weight_grid.csv` | Batch weight grid |
| `weight_*` / `batch_*` / `random_solo_*` | Ablations |
| `*_manifest.json` | git SHA + config snapshot |
| `figures/` | PNG/PDF + CSV tables |

Legacy contention/noise plots: `python -m scheduler.experiment --out scheduler/results`.

## Seeds

- Noise sweep: base seed `42`; trial seed `42 + trial*1000 + int(noise*100)`.
- Deterministic Couloir scores (no RNG in utility).

## Live smoke (optional, not paper-primary)

```bash
cp env.example .env   # GEMINI_API_KEY
python -m scheduler.live_runner --backend gemini --runs 2 --workers 2
```

See [`LIVE_BACKEND.md`](LIVE_BACKEND.md).
