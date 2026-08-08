"""Experiment harness: LRU vs STE vs STE+prefetch on synthetic workflows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from kvflow.cache_sim import DECODE_TOKENS_PER_SEC, PolicyName, PrefixCacheSim
from kvflow.step_graph import AgentStepGraph, steps_to_execution
from kvflow.workflows import research_writing_graph, sequential_agents_graph


@dataclass(frozen=True)
class WorkloadSetting:
    name: str
    fixed_tokens: int
    dynamic_tokens: int
    shared_tokens: int
    capacity_tokens: int
    output_tokens: int


# Tight capacity: shared + two fixed leaves (+ one dynamic) — forces eviction.
# Default graph is a sequential chain so STE retention of the *next* agent
# diverges clearly from LRU (research DAG still available via --graph research).
SETTINGS: dict[str, WorkloadSetting] = {
    "small": WorkloadSetting(
        name="small",
        fixed_tokens=512,
        dynamic_tokens=32,
        shared_tokens=64,
        capacity_tokens=64 + 2 * 512 + 32,
        output_tokens=64,
    ),
    "medium": WorkloadSetting(
        name="medium",
        fixed_tokens=2048,
        dynamic_tokens=64,
        shared_tokens=256,
        capacity_tokens=256 + 2 * 2048 + 64,
        output_tokens=128,
    ),
    "large": WorkloadSetting(
        name="large",
        fixed_tokens=8192,
        dynamic_tokens=32,
        shared_tokens=512,
        capacity_tokens=512 + 2 * 8192 + 32,
        output_tokens=256,
    ),
}


def ready_agents(graph: AgentStepGraph, completed: set[str]) -> list[str]:
    g = graph.to_networkx()
    ready: list[str] = []
    for node in graph.nodes:
        if node in completed:
            continue
        preds = list(g.predecessors(node))
        if all(p in completed for p in preds):
            ready.append(node)
    order = {n: i for i, n in enumerate(graph.nodes)}
    ready.sort(key=lambda n: order[n])
    return ready


def run_one_workflow(
    sim: PrefixCacheSim,
    graph: AgentStepGraph,
    *,
    dynamic_tokens: int,
    output_tokens: int,
) -> float:
    """Execute one DAG pass; return simulated wall time for this pass."""
    start = sim.now
    completed: set[str] = set()
    remaining = set(graph.nodes)

    while remaining:
        wave = ready_agents(graph, completed)
        if not wave:
            raise RuntimeError("Workflow stuck: no ready agents")

        ste = steps_to_execution(graph, active_agents=wave)
        sim.update_ste(ste)
        sim.start_prefetch([a for a, s in ste.items() if s == 1])

        for agent in wave:
            ste_now = steps_to_execution(graph, active_agents=[agent])
            # Keep siblings in the same wave as STE 0 so they are not treated
            # as far-future eviction targets while still ready.
            for sibling in wave:
                ste_now[sibling] = 0
            sim.update_ste(ste_now)
            sim.start_prefetch([a for a, s in ste_now.items() if s == 1])

            sim.access_agent(agent, dynamic_tokens=dynamic_tokens)
            sim.advance_decode(output_tokens)

            completed.add(agent)
            remaining.discard(agent)

    elapsed = sim.now - start
    sim.stats.workflow_latency_s += elapsed
    return elapsed


def run_experiment(
    *,
    setting: WorkloadSetting,
    policy: PolicyName,
    graph: AgentStepGraph,
    iterations: int = 8,
    warmup: int = 2,
) -> dict:
    sim = PrefixCacheSim(
        agents=list(graph.nodes),
        shared_tokens=setting.shared_tokens,
        fixed_tokens=setting.fixed_tokens,
        capacity_tokens=setting.capacity_tokens,
        policy=policy,
    )

    for _ in range(warmup):
        run_one_workflow(
            sim,
            graph,
            dynamic_tokens=setting.dynamic_tokens,
            output_tokens=setting.output_tokens,
        )

    base_hits = sim.stats.hits
    base_misses = sim.stats.misses
    base_recompute = sim.stats.recompute_tokens
    base_transfer = sim.stats.transfer_tokens
    base_stall = sim.stats.stall_s
    base_evict = sim.stats.eviction_count
    base_latency = sim.stats.workflow_latency_s

    measured = max(iterations - warmup, 1)
    for _ in range(measured):
        run_one_workflow(
            sim,
            graph,
            dynamic_tokens=setting.dynamic_tokens,
            output_tokens=setting.output_tokens,
        )

    hits = sim.stats.hits - base_hits
    misses = sim.stats.misses - base_misses
    accesses = hits + misses
    latency = sim.stats.workflow_latency_s - base_latency

    return {
        "setting": setting.name,
        "policy": policy,
        "fixed_tokens": setting.fixed_tokens,
        "dynamic_tokens": setting.dynamic_tokens,
        "capacity_tokens": setting.capacity_tokens,
        "iterations_measured": measured,
        "avg_latency_s": latency / measured,
        "hit_rate": hits / accesses if accesses else 0.0,
        "recompute_tokens": sim.stats.recompute_tokens - base_recompute,
        "transfer_tokens": sim.stats.transfer_tokens - base_transfer,
        "stall_s": sim.stats.stall_s - base_stall,
        "eviction_count": sim.stats.eviction_count - base_evict,
        "decode_tokens_per_sec": DECODE_TOKENS_PER_SEC,
    }


def build_graph(name: str) -> AgentStepGraph:
    if name == "research":
        return research_writing_graph()
    if name.startswith("sequential"):
        parts = name.split(":")
        n = int(parts[1]) if len(parts) > 1 else 10
        return sequential_agents_graph(n)
    raise ValueError(f"Unknown graph {name!r}; use 'research' or 'sequential[:N]'")


def plot_summary(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("avg_latency_s", "Avg simulated latency (s)"),
        ("hit_rate", "Cache hit rate"),
        ("stall_s", "Total stall time (s)"),
        ("eviction_count", "Eviction count"),
    ]
    settings = list(df["setting"].unique())
    policies = list(df["policy"].unique())
    x = range(len(settings))
    width = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (col, title) in zip(axes.flat, metrics):
        for i, policy in enumerate(policies):
            subset = df[df["policy"] == policy].set_index("setting").reindex(settings)
            ax.bar(
                [xi + (i - (len(policies) - 1) / 2) * width for xi in x],
                subset[col].fillna(0.0),
                width=width,
                label=policy,
            )
        ax.set_xticks(list(x))
        ax.set_xticklabels(settings)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("KVFlow cache policies (conceptual sim)")
    fig.tight_layout()
    out_path = out_dir / "cache_policy_comparison.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare LRU / STE / STE+prefetch prefix-cache policies"
    )
    parser.add_argument(
        "--configs",
        default="lru,ste,ste+prefetch",
        help="Comma-separated policies",
    )
    parser.add_argument(
        "--settings",
        default="small,medium,large",
        help="Comma-separated workload sizes",
    )
    parser.add_argument(
        "--graph",
        default="sequential:10",
        help="Workflow graph: research | sequential[:N] (default sequential:10)",
    )
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("kvflow/results"),
        help="Output directory for CSV/plots",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    policies = [p.strip() for p in args.configs.split(",") if p.strip()]
    setting_names = [s.strip() for s in args.settings.split(",") if s.strip()]
    graph = build_graph(args.graph)

    rows = []
    for sname in setting_names:
        if sname not in SETTINGS:
            raise SystemExit(f"Unknown setting {sname!r}; choose from {list(SETTINGS)}")
        setting = SETTINGS[sname]
        for policy in policies:
            if policy not in ("lru", "ste", "ste+prefetch"):
                raise SystemExit(f"Unknown policy {policy!r}")
            row = run_experiment(
                setting=setting,
                policy=policy,  # type: ignore[arg-type]
                graph=graph,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            rows.append(row)
            print(
                f"{sname:8} {policy:14}  "
                f"latency={row['avg_latency_s']:.3f}s  "
                f"hit_rate={row['hit_rate']:.3f}  "
                f"stall={row['stall_s']:.3f}s  "
                f"evict={row['eviction_count']}"
            )

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    if not args.no_plot:
        plot_summary(df, args.out)
        print(f"Saved {args.out / 'cache_policy_comparison.png'}")


if __name__ == "__main__":
    main()
