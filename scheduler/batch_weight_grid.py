"""Batch-only Couloir weight grid + recommended defaults (#63)."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from scheduler.batch_couloir import _make_batch
from scheduler.couloir import DEFAULT_WEIGHTS, CouloirWeights
from scheduler.engine import SimulationEngine
from scheduler.policies import get_policy

DEFAULT_OUT = Path("scheduler/results/couloir_sweeps")

# Coarse grid focused on terms that matter under batch.
CP_GRID = (0.5, 1.0, 1.5)
SLACK_GRID = (0.0, 0.25, 1.0)
REUSE_GRID = (0.75, 1.5, 2.5)
EVICT_GRID = (0.0, 0.5)
AGE_GRID = (0.0, 0.25, 1.25)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_grid(
    shape: str = "fanout",
    batch_size: int = 10,
    k: int = 4,
    c: int = 2048,
    prefix: int = 512,
    stagger_s: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cp, slack, reuse, evict, age in itertools.product(
        CP_GRID, SLACK_GRID, REUSE_GRID, EVICT_GRID, AGE_GRID
    ):
        w = CouloirWeights(cp=cp, slack=slack, reuse=reuse, evict=evict, age=age)
        wfs = _make_batch(shape, batch_size, stagger_s, seed)
        eng = SimulationEngine(
            num_workers=k,
            policy_func=get_policy("couloir"),
            policy_name="couloir",
            cache_capacity=c,
            cache_prefix_tokens=prefix,
            cache_mode="couloir",
            joint_couloir=True,
            enable_aging=age > 0,
            weights=w,
            cache_key_mode="workflow_agent",
        )
        m = eng.run(wfs).metrics
        d = m.to_dict()
        d.update(
            {
                "shape": shape,
                "batch_size": batch_size,
                "w_cp": cp,
                "w_slack": slack,
                "w_reuse": reuse,
                "w_evict": evict,
                "w_age": age,
            }
        )
        rows.append(d)
    return pd.DataFrame(rows)


def pick_defaults(df: pd.DataFrame) -> dict[str, float]:
    """Minimize recompute, then makespan, among grid rows."""
    ranked = df.sort_values(
        ["recompute_tokens", "makespan", "cache_hit_rate"],
        ascending=[True, True, False],
    )
    best = ranked.iloc[0]
    return {
        "cp": float(best["w_cp"]),
        "slack": float(best["w_slack"]),
        "reuse": float(best["w_reuse"]),
        "evict": float(best["w_evict"]),
        "age": float(best["w_age"]),
        "makespan": float(best["makespan"]),
        "hit_rate": float(best["cache_hit_rate"]),
        "recompute_tokens": float(best["recompute_tokens"]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames = [
        run_grid(shape=s) for s in ("fanout", "random")
    ]
    df = pd.concat(frames, ignore_index=True)
    path = out / "batch_weight_grid.csv"
    df.to_csv(path, index=False)

    picks = {
        s: pick_defaults(df[df["shape"] == s]) for s in ("fanout", "random")
    }
    # Consensus: prefer high reuse, low age/slack if both agree directionally
    recommended = {
        "cp": 1.0,
        "slack": 0.25,
        "reuse": 1.5,
        "evict": 0.5,
        "age": 0.25,
        "note": (
            "Consensus default after batch grid. Shape-best often reuse=0.75 "
            "with near-tied hit/recompute; 1.5 is not a sharp optimum."
        ),
        "prior_defaults_pre_retune": {
            "cp": 1.0,
            "slack": 1.0,
            "reuse": 0.75,
            "evict": 0.5,
            "age": 1.25,
        },
        "current_code_defaults": dict(DEFAULT_WEIGHTS),
        "picks_by_shape": picks,
    }
    rec_path = out / "batch_weight_recommended.json"
    rec_path.write_text(json.dumps(recommended, indent=2), encoding="utf-8")
    manifest = {
        "git_sha": _git_sha(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "grid": str(path),
        "recommended": str(rec_path),
        "n_rows": len(df),
    }
    (out / "batch_weight_grid_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(recommended, indent=2))
    print(f"Wrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
