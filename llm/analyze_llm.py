import json
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt



EDGES = [
    ("planner", "researcher_a"),
    ("planner", "researcher_b"),
    ("planner", "researcher_c"),
    ("researcher_a", "synthesizer"),
    ("researcher_b", "synthesizer"),
    ("researcher_c", "synthesizer"),
    ("synthesizer", "critic"),
    ("critic", "final_writer"),
]


def load_traces(path: str = "traces_llm.jsonl"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def critical_path(trace: list[dict]) -> tuple[list[str], float]:
    durations = {event["agent"]: event["runtime_s"] for event in trace}

    g = nx.DiGraph()
    g.add_edges_from(EDGES)

    start_time = {}
    finish_time = {}
    parent = {}

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

    path = []
    current = end_node
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()

    return path, finish_time[end_node]


def compute_waiting_times(trace: list[dict]) -> dict[str, float]:
    by_agent = {event["agent"]: event for event in trace}
    waiting = {}

    for event in trace:
        agent = event["agent"]
        parents = event["parents"]
        if not parents:
            waiting[agent] = 0.0
            continue
        ready_time = max(by_agent[p]["end_time"] for p in parents)
        waiting[agent] = max(0.0, event["start_time"] - ready_time)

    return waiting

def plot_workflow_graph(
    out_path: str = "workflow_graph_llm.png",
    critical_path_nodes: list[str] | None = None,
) -> None:
    """Draw the LangGraph-style agent dependency graph, highlighting the critical path."""
    g = nx.DiGraph()
    g.add_edges_from(EDGES)

    # Manual layout: parallel researchers in the middle
    pos = {
        "planner": (0.0, 0.0),
        "researcher_a": (1.0, 1.0),
        "researcher_b": (1.0, 0.0),
        "researcher_c": (1.0, -1.0),
        "synthesizer": (2.0, 0.0),
        "critic": (3.0, 0.0),
        "final_writer": (4.0, 0.0),
    }

    if critical_path_nodes and len(critical_path_nodes) >= 2:
        critical_edges = list(zip(critical_path_nodes[:-1], critical_path_nodes[1:]))
    else:
        # Fallback: most common path from the LLM profiling runs
        critical_edges = [
            ("planner", "researcher_b"),
            ("researcher_b", "synthesizer"),
            ("synthesizer", "critic"),
            ("critic", "final_writer"),
        ]

    other_edges = [e for e in EDGES if e not in critical_edges]
    critical_nodes = set()
    for u, v in critical_edges:
        critical_nodes.add(u)
        critical_nodes.add(v)
    normal_nodes = [n for n in g.nodes if n not in critical_nodes]

    plt.figure(figsize=(10, 4))
    if normal_nodes:
        nx.draw_networkx_nodes(
            g, pos, nodelist=normal_nodes, node_size=1800, node_color="#dbeafe"
        )
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=list(critical_nodes),
        node_size=1800,
        node_color="#fecaca",
    )
    nx.draw_networkx_labels(g, pos, font_size=8)
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=other_edges,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        width=1.2,
        edge_color="#94a3b8",
    )
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=critical_edges,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        width=2.5,
        edge_color="#dc2626",
    )
    plt.title("Multi-Agent Workflow Graph (red = most common critical path)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_runtime_chart(df: pd.DataFrame, out_path: str = "avg_runtime.png") -> None:
    plot_df = df.sort_values("avg_runtime_s", ascending=True)

    plt.figure(figsize=(8, 4.5))
    plt.barh(plot_df["agent"], plot_df["avg_runtime_s"], color="#60a5fa")
    plt.xlabel("Average runtime (seconds)")
    plt.title("Average Agent Runtime (LLM runs)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_critical_freq_chart(
    df: pd.DataFrame, out_path: str = "critical_path_freq.png"
) -> None:
    plot_df = df.sort_values("critical_path_freq", ascending=True)

    plt.figure(figsize=(8, 4.5))
    plt.barh(plot_df["agent"], plot_df["critical_path_freq"], color="#34d399")
    plt.xlabel("Critical-path frequency")
    plt.xlim(0, 1.05)
    plt.title("How Often Each Agent Lies on the Critical Path")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def main():
    rows = load_traces("traces_llm.jsonl")

    runtime_sum = defaultdict(float)
    runtime_count = defaultdict(int)
    waiting_sum = defaultdict(float)
    critical_counts = Counter()
    path_counts = Counter()
    latencies = []

    for row in rows:
        trace = row["trace"]
        path, latency = critical_path(trace)
        latencies.append(latency)
        path_counts[tuple(path)] += 1

        for agent in path:
            critical_counts[agent] += 1

        waiting = compute_waiting_times(trace)

        for event in trace:
            agent = event["agent"]
            runtime_sum[agent] += event["runtime_s"]
            runtime_count[agent] += 1
            waiting_sum[agent] += waiting.get(agent, 0.0)

    agents = sorted(runtime_count.keys())
    records = []

    for agent in agents:
        records.append(
            {
                "agent": agent,
                "avg_runtime_s": runtime_sum[agent] / runtime_count[agent],
                "avg_waiting_s": waiting_sum[agent] / runtime_count[agent],
                "critical_path_count": critical_counts[agent],
                "critical_path_freq": critical_counts[agent] / len(rows),
            }
        )

    df = pd.DataFrame(records).sort_values("critical_path_freq", ascending=False)

    print(f"Runs analyzed: {len(rows)}")
    print(f"Average end-to-end latency (critical-path length): {sum(latencies)/len(latencies):.3f}s\n")
    print(df.to_string(index=False))

    most_common_path = list(path_counts.most_common(1)[0][0])
    print(f"\nMost common critical path: {' -> '.join(most_common_path)}")

    df.to_csv("profiling_summary_llm.csv", index=False)
    print("\nSaved profiling_summary_llm.csv")

    plot_workflow_graph("workflow_graph_llm.png", critical_path_nodes=most_common_path)
    plot_runtime_chart(df, "avg_runtime.png")
    plot_critical_freq_chart(df, "critical_path_freq.png")


if __name__ == "__main__":
    main()