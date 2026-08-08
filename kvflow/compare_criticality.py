"""Compare topology STE with runtime-weighted critical-path frequency."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from kvflow.step_graph import steps_to_execution
from kvflow.workflows import RESEARCH_WRITING_EDGES, research_writing_graph


def load_traces(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def critical_path(trace: list[dict], edges: list[tuple[str, str]]) -> tuple[list[str], float]:
    """Longest runtime-weighted path (same algorithm as simulated/analyze.py)."""
    durations = {event["agent"]: event["runtime_s"] for event in trace}

    g = nx.DiGraph()
    g.add_edges_from(edges)

    start_time: dict[str, float] = {}
    finish_time: dict[str, float] = {}
    parent: dict[str, str | None] = {}

    for node in nx.topological_sort(g):
        preds = list(g.predecessors(node))
        if not preds:
            start_time[node] = 0.0
            parent[node] = None
        else:
            best_parent = max(preds, key=lambda p: finish_time[p])
            start_time[node] = finish_time[best_parent]
            parent[node] = best_parent
        finish_time[node] = start_time[node] + durations.get(node, 0.0)

    end_node = max(finish_time, key=finish_time.get)
    path: list[str] = []
    current: str | None = end_node
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path, finish_time[end_node]


def analyze_traces(rows: list[dict]) -> pd.DataFrame:
    graph = research_writing_graph()
    ste = steps_to_execution(graph, active_agents=["planner"])

    runtime_sum: dict[str, float] = defaultdict(float)
    runtime_count: dict[str, int] = defaultdict(int)
    critical_counts: Counter[str] = Counter()

    for row in rows:
        trace = row["trace"]
        path, _ = critical_path(trace, RESEARCH_WRITING_EDGES)
        for agent in path:
            critical_counts[agent] += 1
        for event in trace:
            agent = event["agent"]
            runtime_sum[agent] += event["runtime_s"]
            runtime_count[agent] += 1

    n_runs = max(len(rows), 1)
    records = []
    agents = sorted(set(runtime_count) | set(graph.nodes))
    for agent in agents:
        records.append(
            {
                "agent": agent,
                "avg_runtime_s": (
                    runtime_sum[agent] / runtime_count[agent]
                    if runtime_count[agent]
                    else 0.0
                ),
                "critical_path_freq": critical_counts[agent] / n_runs,
                "steps_to_execution": ste.get(agent),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["steps_to_execution", "agent"], ascending=[True, True]
    )


def plot_comparison(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df.sort_values("steps_to_execution")
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    x = range(len(plot_df))
    ax1.bar(
        [i - 0.2 for i in x],
        plot_df["steps_to_execution"],
        width=0.4,
        label="STE (topology)",
        color="#60a5fa",
    )
    ax1.set_ylabel("Steps-to-execution")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(plot_df["agent"], rotation=30, ha="right")

    ax2 = ax1.twinx()
    ax2.bar(
        [i + 0.2 for i in x],
        plot_df["critical_path_freq"],
        width=0.4,
        label="Critical-path freq",
        color="#34d399",
    )
    ax2.set_ylabel("Critical-path frequency")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("STE (structure-only) vs critical-path frequency (latency-weighted)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare steps-to-execution with critical-path frequency"
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path("simulated/traces.jsonl"),
        help="Path to workflow traces JSONL",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("kvflow/results"),
        help="Output directory for CSV/plot",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing the comparison plot",
    )
    args = parser.parse_args(argv)

    if not args.traces.exists():
        # Fall back to topology-only STE table when traces are absent.
        graph = research_writing_graph()
        ste = steps_to_execution(graph, active_agents=["planner"])
        df = pd.DataFrame(
            [
                {
                    "agent": agent,
                    "avg_runtime_s": float("nan"),
                    "critical_path_freq": float("nan"),
                    "steps_to_execution": ste[agent],
                }
                for agent in graph.nodes
            ]
        )
        print(f"No traces at {args.traces}; emitting STE-only table.\n")
    else:
        rows = load_traces(args.traces)
        df = analyze_traces(rows)
        print(f"Runs analyzed: {len(rows)}")
        print(
            "Note: STE is structure-only (when each agent runs); "
            "critical-path frequency is latency-weighted from measured runtimes.\n"
        )

    print(df.to_string(index=False))
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "ste_vs_critical_path.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    if not args.no_plot and df["critical_path_freq"].notna().any():
        plot_path = args.out / "ste_vs_critical_path.png"
        plot_comparison(df, plot_path)
        print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
