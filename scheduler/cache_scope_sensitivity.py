"""Compare agent_type vs workflow_agent cache keying (#62)."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from scheduler.batch_couloir import _make_batch
from scheduler.engine import SimulationEngine
from scheduler.policies import get_policy

DEFAULT_OUT = Path("scheduler/results/couloir_sweeps")
CONFIGS = (
    ("fifo", "lru"),
    ("cpf", "ste"),
    ("two_phase", "ste"),
    ("couloir", "couloir"),
)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_scope_matrix(
    shapes: Sequence[str] = ("fanout", "random", "writing"),
    batch_sizes: Sequence[int] = (5, 10, 20),
    key_modes: Sequence[str] = ("agent_type", "workflow_agent"),
    num_workers: int = 4,
    cache_capacity: int = 2048,
    prefix_tokens: int = 512,
    stagger_s: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for shape in shapes:
        for m in batch_sizes:
            for key_mode in key_modes:
                for pol, mode in CONFIGS:
                    wfs = _make_batch(shape, m, stagger_s, seed)
                    eng = SimulationEngine(
                        num_workers=num_workers,
                        policy_func=get_policy(pol),
                        policy_name=pol,
                        cache_capacity=cache_capacity,
                        cache_prefix_tokens=prefix_tokens,
                        cache_mode=mode,
                        joint_couloir=(mode == "couloir"),
                        cache_key_mode=key_mode,
                    )
                    res = eng.run(wfs)
                    d = res.metrics.to_dict()
                    d.update(
                        {
                            "shape": shape,
                            "batch_size": m,
                            "cache_key_mode": key_mode,
                            "policy": pol,
                            "cache_mode": mode,
                            "K": num_workers,
                            "C": cache_capacity,
                        }
                    )
                    rows.append(d)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = run_scope_matrix()
    path = out / "cache_scope_sensitivity.csv"
    df.to_csv(path, index=False)
    summary = (
        df.groupby(["shape", "batch_size", "cache_key_mode", "policy"], as_index=False)
        .agg(
            makespan=("makespan", "mean"),
            hit_rate=("cache_hit_rate", "mean"),
            recompute=("recompute_tokens", "mean"),
        )
    )
    sum_path = out / "cache_scope_summary.csv"
    summary.to_csv(sum_path, index=False)
    manifest = {
        "git_sha": _git_sha(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "csv": str(path),
        "summary": str(sum_path),
        "default_engine_mode": "workflow_agent",
    }
    (out / "cache_scope_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    # Print Couloir vs CPF recompute gap under scoped keys
    scoped = summary[
        (summary.cache_key_mode == "workflow_agent") & (summary.batch_size == 20)
    ]
    print(scoped[scoped.policy.isin(["couloir", "cpf", "fifo"])].to_string(index=False))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
