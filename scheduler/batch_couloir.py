"""Batch / multi-arrival Couloir evaluation on named + random DAGs (#59)."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from scheduler.engine import SimulationEngine
from scheduler.policies import get_policy
from scheduler.random_dags import create_random_layered_dag
from scheduler.workloads import create_workflow

DEFAULT_OUT = Path("scheduler/results/couloir_sweeps")

CONFIGS: list[tuple[str, str]] = [
    ("fifo", "lru"),
    ("cpf", "ste"),
    ("two_phase", "ste"),
    ("couloir", "couloir"),
]


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _make_batch(
    shape: str,
    m: int,
    stagger_s: float,
    seed: int,
    *,
    n_nodes: int = 8,
    n_layers: int = 4,
) -> list:
    wfs = []
    for i in range(m):
        arrival = i * stagger_s
        if shape == "random":
            wfs.append(
                create_random_layered_dag(
                    workflow_id=f"wf_{i}",
                    n_nodes=n_nodes,
                    n_layers=n_layers,
                    arrival_time=arrival,
                    rng=random.Random(seed + i * 31),
                )
            )
        else:
            wfs.append(
                create_workflow(
                    shape,  # type: ignore[arg-type]
                    workflow_id=f"wf_{i}",
                    profile="llm",
                    arrival_time=arrival,
                )
            )
    return wfs


def run_batch_matrix(
    shapes: Sequence[str] = ("writing", "fanout", "random"),
    batch_sizes: Sequence[int] = (1, 5, 10, 20),
    num_workers: int = 4,
    cache_capacity: int = 2048,
    prefix_tokens: int = 512,
    stagger_s: float = 2.0,
    seed: int = 42,
    aging_modes: Sequence[bool] = (True, False),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for shape in shapes:
        for m in batch_sizes:
            for aging in aging_modes:
                for pol, mode in CONFIGS:
                    if pol != "couloir" and not aging:
                        # Aging only meaningful for Couloir; skip duplicate baselines.
                        continue
                    wfs = _make_batch(shape, m, stagger_s, seed)
                    eng = SimulationEngine(
                        num_workers=num_workers,
                        policy_func=get_policy(pol),
                        policy_name=pol,
                        cache_capacity=cache_capacity,
                        cache_prefix_tokens=prefix_tokens,
                        cache_mode=mode,
                        joint_couloir=(mode == "couloir"),
                        enable_aging=aging if pol == "couloir" else True,
                    )
                    res = eng.run(wfs)
                    d = res.metrics.to_dict()
                    d.update(
                        {
                            "shape": shape,
                            "batch_size": m,
                            "K": num_workers,
                            "C": cache_capacity,
                            "prefix_tokens": prefix_tokens,
                            "stagger_s": stagger_s,
                            "seed": seed,
                            "enable_aging": aging if pol == "couloir" else None,
                            "policy": pol,
                            "cache_mode": mode,
                        }
                    )
                    rows.append(d)
    return pd.DataFrame(rows)


def run_random_single_family(
    n_dags: int = 24,
    num_workers: int = 2,
    cache_capacity: int = 2048,
    prefix_tokens: int = 512,
    seed: int = 42,
) -> pd.DataFrame:
    """One random DAG at a time; report distribution across seeds."""
    rows: list[dict[str, Any]] = []
    for i in range(n_dags):
        for pol, mode in CONFIGS:
            wf = create_random_layered_dag(
                workflow_id=f"solo_{i}",
                arrival_time=0.0,
                rng=random.Random(seed + i * 17),
            )
            eng = SimulationEngine(
                num_workers=num_workers,
                policy_func=get_policy(pol),
                policy_name=pol,
                cache_capacity=cache_capacity,
                cache_prefix_tokens=prefix_tokens,
                cache_mode=mode,
                joint_couloir=(mode == "couloir"),
            )
            res = eng.run([wf])
            d = res.metrics.to_dict()
            d.update(
                {
                    "dag_index": i,
                    "seed": seed + i * 17,
                    "K": num_workers,
                    "C": cache_capacity,
                    "policy": pol,
                    "cache_mode": mode,
                    "experiment": "random_solo",
                }
            )
            rows.append(d)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Batch + random Couloir evaluation")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    batch = run_batch_matrix(seed=args.seed)
    solo = run_random_single_family(seed=args.seed)
    batch_path = out / "batch_multi_arrival.csv"
    solo_path = out / "random_solo_matrix.csv"
    batch.to_csv(batch_path, index=False)
    solo.to_csv(solo_path, index=False)

    # Summary: mean makespan / hit rate by shape×batch×policy (aging on).
    sub = batch[batch["enable_aging"].fillna(True) != False]
    summary = (
        sub.groupby(["shape", "batch_size", "policy"], as_index=False)
        .agg(
            makespan=("makespan", "mean"),
            hit_rate=("cache_hit_rate", "mean"),
            recompute=("recompute_tokens", "mean"),
            jct_p50=("jct_p50", "mean"),
        )
    )
    summary_path = out / "batch_summary.csv"
    summary.to_csv(summary_path, index=False)

    solo_sum = (
        solo.groupby(["policy"], as_index=False)
        .agg(
            mean_makespan=("makespan", "mean"),
            std_makespan=("makespan", "std"),
            mean_hit=("cache_hit_rate", "mean"),
            mean_recompute=("recompute_tokens", "mean"),
            n=("makespan", "count"),
        )
    )
    solo_sum_path = out / "random_solo_summary.csv"
    solo_sum.to_csv(solo_sum_path, index=False)

    manifest = {
        "git_sha": _git_sha(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "batch": str(batch_path),
        "random_solo": str(solo_path),
        "batch_summary": str(summary_path),
        "random_solo_summary": str(solo_sum_path),
        "seed": args.seed,
    }
    (out / "batch_random_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote {batch_path} ({len(batch)} rows)")
    print(f"Wrote {solo_path} ({len(solo)} rows)")
    print(solo_sum.to_string(index=False))


if __name__ == "__main__":
    main()
