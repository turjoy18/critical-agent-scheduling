"""Couloir vs baselines sweeps over K, C (issues #19 / #20 / #21)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from scheduler.engine import SimulationEngine
from scheduler.policies import get_policy
from scheduler.workloads import DagName, create_workflow

DEFAULT_OUT = Path("scheduler/results/couloir_sweeps")

# (policy_name, cache_mode)
CONFIGS: list[tuple[str, str]] = [
    ("fifo", "none"),
    ("cpf", "none"),
    ("heft", "none"),
    ("fifo", "lru"),
    ("cpf", "ste"),
    ("two_phase", "ste"),
    ("couloir", "couloir"),
]


def run_matrix(
    dag: DagName = "writing",
    worker_counts: Sequence[int] = (1, 2, 3),
    cache_capacities: Sequence[int] = (40, 80, 160),
    prefix_tokens: int = 20,
    profile: str = "llm",
) -> pd.DataFrame:
    rows: list[dict] = []
    for k in worker_counts:
        for pol, mode in CONFIGS:
            caps: Sequence[int | None]
            if mode == "none":
                caps = [None]
            else:
                caps = list(cache_capacities)
            for cap in caps:
                wf = create_workflow(dag, workflow_id="wf0", profile=profile)
                engine = SimulationEngine(
                    num_workers=k,
                    policy_func=get_policy(pol),
                    policy_name=pol,
                    cache_capacity=cap,
                    cache_prefix_tokens=prefix_tokens,
                    cache_mode=mode,
                    joint_couloir=(mode == "couloir"),
                )
                res = engine.run([wf])
                d = res.metrics.to_dict()
                d.update(
                    {
                        "dag": dag,
                        "K": k,
                        "C": cap if cap is not None else -1,
                        "cache_mode": mode,
                        "prefix_tokens": prefix_tokens,
                    }
                )
                rows.append(d)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Couloir baseline matrix sweeps")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--dag", choices=["writing", "chain", "fanout"], default="writing")
    p.add_argument("--all-dags", action="store_true")
    p.add_argument("--profile", choices=["llm", "simulated"], default="llm")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dags: list[DagName] = (
        ["writing", "chain", "fanout"] if args.all_dags else [args.dag]  # type: ignore[list-item]
    )
    df = pd.concat(
        [run_matrix(dag=d, profile=args.profile) for d in dags], ignore_index=True
    )
    csv_path = out / "couloir_baseline_matrix.csv"
    df.to_csv(csv_path, index=False)
    manifest = {
        "dags": dags,
        "configs": CONFIGS,
        "rows": len(df),
        "csv": str(csv_path),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
