"""Couloir vs baselines sweeps over K, C, aging, noise (#19 / #20 / #21 / #27)."""

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


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def write_manifest(out: Path, payload: dict[str, Any]) -> None:
    body = {
        "git_sha": _git_sha(),
        "timestamp_unix": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    }
    (out / "manifest.json").write_text(json.dumps(body, indent=2), encoding="utf-8")


def run_matrix(
    dag: DagName = "writing",
    worker_counts: Sequence[int] = (1, 2, 3),
    cache_capacities: Sequence[int] = (1024, 2048, 4096),
    prefix_tokens: int = 512,
    profile: str = "llm",
    enable_aging: bool = True,
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
                    enable_aging=enable_aging if mode == "couloir" else True,
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
                        "enable_aging": enable_aging if mode == "couloir" else None,
                        "noise_std": 0.0,
                        "seed": None,
                    }
                )
                rows.append(d)
    return pd.DataFrame(rows)


def run_aging_ablation(
    dag: DagName = "fanout",
    num_workers: int = 2,
    cache_capacity: int = 2048,
    prefix_tokens: int = 512,
    profile: str = "llm",
) -> pd.DataFrame:
    """Couloir with vs without aging under contention."""
    rows: list[dict] = []
    for aging in (True, False):
        wf = create_workflow(dag, workflow_id="wf0", profile=profile)
        engine = SimulationEngine(
            num_workers=num_workers,
            policy_func=get_policy("couloir"),
            policy_name="couloir",
            cache_capacity=cache_capacity,
            cache_prefix_tokens=prefix_tokens,
            cache_mode="couloir",
            joint_couloir=True,
            enable_aging=aging,
        )
        res = engine.run([wf])
        d = res.metrics.to_dict()
        d.update(
            {
                "dag": dag,
                "K": num_workers,
                "C": cache_capacity,
                "cache_mode": "couloir",
                "prefix_tokens": prefix_tokens,
                "enable_aging": aging,
                "policy": "couloir",
            }
        )
        rows.append(d)
    return pd.DataFrame(rows)


def run_noise_sweep(
    dag: DagName = "writing",
    num_workers: int = 2,
    cache_capacity: int = 2048,
    prefix_tokens: int = 512,
    noise_levels: Sequence[float] = (0.0, 0.1, 0.25, 0.5),
    trials: int = 5,
    seed: int = 42,
    profile: str = "llm",
) -> pd.DataFrame:
    """Stochastic duration noise for Couloir vs CPF+STE vs FIFO+LRU."""
    configs = [
        ("fifo", "lru"),
        ("cpf", "ste"),
        ("two_phase", "ste"),
        ("couloir", "couloir"),
    ]
    rows: list[dict] = []
    for noise in noise_levels:
        for trial in range(trials):
            trial_seed = seed + trial * 1000 + int(noise * 100)
            for pol, mode in configs:
                wf = create_workflow(
                    dag,
                    workflow_id="wf0",
                    profile=profile,
                    duration_noise_std=noise,
                    rng=random.Random(trial_seed),
                )
                engine = SimulationEngine(
                    num_workers=num_workers,
                    policy_func=get_policy(pol),
                    policy_name=pol,
                    cache_capacity=cache_capacity,
                    cache_prefix_tokens=prefix_tokens,
                    cache_mode=mode,
                    joint_couloir=(mode == "couloir"),
                )
                res = engine.run([wf])
                d = res.metrics.to_dict()
                d.update(
                    {
                        "dag": dag,
                        "K": num_workers,
                        "C": cache_capacity,
                        "cache_mode": mode,
                        "prefix_tokens": prefix_tokens,
                        "noise_std": noise,
                        "trial": trial,
                        "seed": trial_seed,
                        "policy": pol,
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
    p.add_argument("--with-ablations", action="store_true", help="Aging + noise sweeps")
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

    extra: dict[str, Any] = {}
    if args.with_ablations:
        aging = pd.concat(
            [
                run_aging_ablation(dag=d, profile=args.profile)
                for d in ("writing", "fanout")
            ],
            ignore_index=True,
        )
        aging_path = out / "aging_ablation.csv"
        aging.to_csv(aging_path, index=False)
        extra["aging_ablation"] = str(aging_path)

        noise = pd.concat(
            [
                run_noise_sweep(dag=d, profile=args.profile)
                for d in ("writing", "fanout")
            ],
            ignore_index=True,
        )
        noise_path = out / "noise_sweep.csv"
        noise.to_csv(noise_path, index=False)
        extra["noise_sweep"] = str(noise_path)

    write_manifest(
        out,
        {
            "dags": dags,
            "configs": CONFIGS,
            "rows": len(df),
            "csv": str(csv_path),
            "prefix_tokens": 512,
            "cache_capacities": [1024, 2048, 4096],
            "worker_counts": [1, 2, 3],
            "profile": args.profile,
            **extra,
        },
    )
    print(f"Wrote {csv_path} ({len(df)} rows)")
    if extra:
        print(f"Ablations: {list(extra.values())}")


if __name__ == "__main__":
    main()
