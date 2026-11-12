"""Couloir weight sensitivity + leave-one-term ablation (#58)."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from scheduler.couloir import DEFAULT_WEIGHTS, CouloirWeights
from scheduler.engine import SimulationEngine
from scheduler.policies import get_policy
from scheduler.workloads import DagName, create_workflow

DEFAULT_OUT = Path("scheduler/results/couloir_sweeps")
TERMS = ("cp", "slack", "reuse", "evict", "age")
SCALE_GRID = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _run_one(
    dag: DagName,
    k: int,
    c: int,
    prefix: int,
    weights: CouloirWeights,
    *,
    enable_aging: bool = True,
    policy: str = "couloir",
    cache_mode: str = "couloir",
) -> dict[str, Any]:
    wf = create_workflow(dag, workflow_id="wf0", profile="llm")
    eng = SimulationEngine(
        num_workers=k,
        policy_func=get_policy(policy),
        policy_name=policy,
        cache_capacity=c if cache_mode != "none" else None,
        cache_prefix_tokens=prefix,
        cache_mode=cache_mode,
        joint_couloir=(cache_mode == "couloir"),
        enable_aging=enable_aging,
        weights=weights if cache_mode == "couloir" else None,
    )
    m = eng.run([wf]).metrics
    d = m.to_dict()
    d.update(
        {
            "dag": dag,
            "K": k,
            "C": c,
            "prefix_tokens": prefix,
            "policy": policy,
            "cache_mode": cache_mode,
            "enable_aging": enable_aging,
            **{f"w_{t}": getattr(weights, t) for t in TERMS},
        }
    )
    return d


def run_leave_one_out(
    dags: Sequence[DagName] = ("writing", "fanout"),
    k: int = 2,
    c: int = 2048,
    prefix: int = 512,
) -> pd.DataFrame:
    base = CouloirWeights()
    rows: list[dict[str, Any]] = []
    for dag in dags:
        rows.append(
            {
                **_run_one(dag, k, c, prefix, base),
                "ablation": "default",
                "dropped_term": None,
            }
        )
        for term in TERMS:
            w = replace(base, **{term: 0.0})
            rows.append(
                {
                    **_run_one(dag, k, c, prefix, w, enable_aging=(term != "age")),
                    "ablation": "leave_one_out",
                    "dropped_term": term,
                }
            )
        # Baselines for context
        for pol, mode in (("fifo", "lru"), ("cpf", "ste"), ("two_phase", "ste")):
            rows.append(
                {
                    **_run_one(
                        dag, k, c, prefix, base, policy=pol, cache_mode=mode
                    ),
                    "ablation": "baseline",
                    "dropped_term": None,
                }
            )
    return pd.DataFrame(rows)


def run_sensitivity(
    dags: Sequence[DagName] = ("writing", "fanout"),
    k: int = 2,
    c: int = 2048,
    prefix: int = 512,
    scales: Sequence[float] = SCALE_GRID,
) -> pd.DataFrame:
    base = CouloirWeights()
    rows: list[dict[str, Any]] = []
    for dag in dags:
        for term in TERMS:
            default_v = float(DEFAULT_WEIGHTS[term])
            for scale in scales:
                w = replace(base, **{term: default_v * scale})
                rows.append(
                    {
                        **_run_one(
                            dag,
                            k,
                            c,
                            prefix,
                            w,
                            enable_aging=(term != "age" or scale > 0),
                        ),
                        "ablation": "sensitivity",
                        "varied_term": term,
                        "scale": scale,
                    }
                )
    return pd.DataFrame(rows)


def run_batch_leave_one_out(
    shapes: Sequence[str] = ("fanout", "random", "writing"),
    batch_size: int = 10,
    k: int = 4,
    c: int = 2048,
    prefix: int = 512,
    stagger_s: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Leave-one-out under multi-arrival — where reuse/age can matter."""
    from scheduler.batch_couloir import _make_batch

    base = CouloirWeights()
    rows: list[dict[str, Any]] = []
    variants: list[tuple[str, CouloirWeights, bool]] = [("default", base, True)]
    for term in TERMS:
        variants.append((term, replace(base, **{term: 0.0}), term != "age"))
    for shape in shapes:
        for dropped, weights, aging in variants:
            wfs = _make_batch(shape, batch_size, stagger_s, seed)
            eng = SimulationEngine(
                num_workers=k,
                policy_func=get_policy("couloir"),
                policy_name="couloir",
                cache_capacity=c,
                cache_prefix_tokens=prefix,
                cache_mode="couloir",
                joint_couloir=True,
                enable_aging=aging,
                weights=weights,
            )
            m = eng.run(wfs).metrics
            d = m.to_dict()
            d.update(
                {
                    "shape": shape,
                    "batch_size": batch_size,
                    "K": k,
                    "C": c,
                    "ablation": "batch_leave_one_out",
                    "dropped_term": None if dropped == "default" else dropped,
                    **{f"w_{t}": getattr(weights, t) for t in TERMS},
                }
            )
            rows.append(d)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Couloir weight ablations")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    loo = run_leave_one_out()
    sens = run_sensitivity()
    batch_loo = run_batch_leave_one_out()
    loo_path = out / "weight_leave_one_out.csv"
    sens_path = out / "weight_sensitivity.csv"
    batch_loo_path = out / "weight_batch_leave_one_out.csv"
    loo.to_csv(loo_path, index=False)
    sens.to_csv(sens_path, index=False)
    batch_loo.to_csv(batch_loo_path, index=False)

    manifest = {
        "git_sha": _git_sha(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "default_weights": dict(DEFAULT_WEIGHTS),
        "leave_one_out": str(loo_path),
        "sensitivity": str(sens_path),
        "batch_leave_one_out": str(batch_loo_path),
        "scales": list(SCALE_GRID),
    }
    (out / "weight_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote {loo_path} ({len(loo)} rows)")
    print(f"Wrote {sens_path} ({len(sens)} rows)")
    print(f"Wrote {batch_loo_path} ({len(batch_loo)} rows)")


if __name__ == "__main__":
    main()
