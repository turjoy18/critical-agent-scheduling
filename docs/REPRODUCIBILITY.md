# Reproducibility

## One-command paper artifacts

```bash
python -m unittest discover -s scheduler/tests -v
python -m scheduler.couloir_sweep --all-dags --with-ablations
python -m scheduler.weight_ablation
python -m scheduler.batch_couloir
python -m scheduler.plot_couloir
```

Outputs under `scheduler/results/couloir_sweeps/`:

| File | Contents |
|------|----------|
| `couloir_baseline_matrix.csv` | \(K\times C\) × policies × DAGs |
| `aging_ablation.csv` | Couloir aging on/off |
| `noise_sweep.csv` | Duration noise trials |
| `weight_leave_one_out.csv` / `weight_sensitivity.csv` | Single-WF weight ablations |
| `weight_batch_leave_one_out.csv` | Batch \(M=10\) leave-one-out |
| `batch_multi_arrival.csv` / `random_solo_*.csv` | Multi-arrival + random DAGs |
| `manifest.json` | git SHA, timestamp, config snapshot |
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
