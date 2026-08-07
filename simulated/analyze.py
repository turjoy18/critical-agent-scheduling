import json
from collections import Counter, defaultdict

import networkx as nx
import pandas as pd


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


def load_traces(path: str = "traces.jsonl"):
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


def main():
    rows = load_traces("traces.jsonl")

    runtime_sum = defaultdict(float)
    runtime_count = defaultdict(int)
    waiting_sum = defaultdict(float)
    critical_counts = Counter()
    latencies = []

    for row in rows:
        trace = row["trace"]
        path, latency = critical_path(trace)
        latencies.append(latency)

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

    df.to_csv("profiling_summary.csv", index=False)
    print("\nSaved profiling_summary.csv")

    
if __name__ == "__main__":
    main()