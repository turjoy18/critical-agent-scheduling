"""Regenerate Couloir paper figures/tables from sweep CSVs (#26)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_CSV = Path("scheduler/results/couloir_sweeps/couloir_baseline_matrix.csv")
DEFAULT_OUT = Path("scheduler/results/couloir_sweeps/figures")

LABEL = {
    ("fifo", "none"): "FIFO",
    ("cpf", "none"): "CPF",
    ("heft", "none"): "HEFT",
    ("fifo", "lru"): "FIFO+LRU",
    ("cpf", "ste"): "CPF+STE",
    ("two_phase", "ste"): "2P+STE",
    ("couloir", "couloir"): "Couloir",
}


def _label(row: pd.Series) -> str:
    return LABEL.get((row["policy"], row["cache_mode"]), f"{row['policy']}+{row['cache_mode']}")


def load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df["label"] = df.apply(_label, axis=1)
    return df


def fig_makespan_vs_k(df: pd.DataFrame, out: Path) -> None:
    """Makespan vs K for cache-off policies + Couloir at mid C."""
    dags = sorted(df["dag"].unique())
    fig, axes = plt.subplots(1, len(dags), figsize=(4.2 * len(dags), 3.6), sharey=True)
    if len(dags) == 1:
        axes = [axes]
    mid_c = sorted(c for c in df["C"].unique() if c > 0)
    mid = mid_c[len(mid_c) // 2] if mid_c else 2048
    focus = df[
        ((df["cache_mode"] == "none") & df["policy"].isin(["fifo", "cpf", "heft"]))
        | ((df["policy"] == "couloir") & (df["C"] == mid))
    ]
    for ax, dag in zip(axes, dags):
        sub = focus[focus["dag"] == dag]
        for label, g in sub.groupby("label", sort=False):
            g = g.sort_values("K")
            ax.plot(g["K"], g["makespan"], marker="o", label=label)
        ax.set_title(dag)
        ax.set_xlabel("K workers")
        ax.set_xticks(sorted(sub["K"].unique()))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Makespan")
    axes[-1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out / "makespan_vs_k.png", dpi=160)
    fig.savefig(out / "makespan_vs_k.pdf")
    plt.close(fig)


def fig_hit_rate_vs_c(df: pd.DataFrame, out: Path) -> None:
    """Cache hit rate vs C for cache-aware configs (K=2)."""
    sub = df[(df["K"] == 2) & (df["C"] > 0)].copy()
    dags = sorted(sub["dag"].unique())
    fig, axes = plt.subplots(1, len(dags), figsize=(4.2 * len(dags), 3.6), sharey=True)
    if len(dags) == 1:
        axes = [axes]
    for ax, dag in zip(axes, dags):
        gdag = sub[sub["dag"] == dag]
        for label, g in gdag.groupby("label", sort=False):
            g = g.sort_values("C")
            ax.plot(g["C"], g["cache_hit_rate"], marker="o", label=label)
        ax.set_title(f"{dag} (K=2)")
        ax.set_xlabel("Cache capacity C")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Cache hit rate")
    axes[-1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out / "hit_rate_vs_c.png", dpi=160)
    fig.savefig(out / "hit_rate_vs_c.pdf")
    plt.close(fig)


def fig_recompute_vs_policy(df: pd.DataFrame, out: Path) -> None:
    """Recompute tokens at K=2 and mid cache capacity."""
    mid_c = sorted(c for c in df["C"].unique() if c > 0)
    mid = mid_c[len(mid_c) // 2] if mid_c else 2048
    sub = df[(df["K"] == 2) & (df["C"] == mid)].copy()
    if sub.empty:
        return
    dags = sorted(sub["dag"].unique())
    fig, axes = plt.subplots(1, len(dags), figsize=(4.2 * len(dags), 3.6), sharey=True)
    if len(dags) == 1:
        axes = [axes]
    for ax, dag in zip(axes, dags):
        g = sub[sub["dag"] == dag].sort_values("recompute_tokens", ascending=True)
        ax.barh(g["label"], g["recompute_tokens"])
        ax.set_title(f"{dag} (K=2, C={mid})")
        ax.grid(True, axis="x", alpha=0.3)
    axes[0].set_xlabel("Recompute tokens")
    fig.tight_layout()
    fig.savefig(out / "recompute_tokens.png", dpi=160)
    fig.savefig(out / "recompute_tokens.pdf")
    plt.close(fig)


def write_tables(df: pd.DataFrame, out: Path) -> None:
    mid_c = sorted(c for c in df["C"].unique() if c > 0)
    mid = mid_c[len(mid_c) // 2] if mid_c else 2048
    pivot = (
        df[df["C"].isin([-1, mid])]
        .assign(cfg=lambda x: x["label"] + " C=" + x["C"].astype(str))
        .pivot_table(index=["dag", "K"], columns="cfg", values="makespan", aggfunc="first")
    )
    pivot.to_csv(out / "makespan_table.csv")
    hits = (
        df[df["C"] > 0]
        .pivot_table(
            index=["dag", "K", "C"],
            columns="label",
            values=["cache_hit_rate", "recompute_tokens"],
            aggfunc="first",
        )
    )
    hits.to_csv(out / "cache_metrics_table.csv")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Couloir sweep figures")
    p.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(Path(args.csv))
    fig_makespan_vs_k(df, out)
    fig_hit_rate_vs_c(df, out)
    fig_recompute_vs_policy(df, out)
    write_tables(df, out)
    print(f"Wrote figures + tables under {out}")


if __name__ == "__main__":
    main()
