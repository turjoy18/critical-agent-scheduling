"""Random DAG generators for Couloir breadth eval (#59)."""

from __future__ import annotations

import random

from scheduler.dag import Task, WorkflowDAG

# Shared agent-type vocabulary so batch runs can reuse prefixes across workflows.
AGENT_POOL = [
    "planner",
    "researcher_a",
    "researcher_b",
    "researcher_c",
    "analyst",
    "critic",
    "synthesizer",
    "writer",
    "verifier",
    "joiner",
]


def create_random_layered_dag(
    workflow_id: str,
    *,
    n_nodes: int = 8,
    n_layers: int = 4,
    edge_prob: float = 0.55,
    duration_lo: float = 5.0,
    duration_hi: float = 30.0,
    arrival_time: float = 0.0,
    rng: random.Random | None = None,
) -> WorkflowDAG:
    """Layered random DAG: edges only forward across layers (guaranteed acyclic)."""
    if n_nodes < 2:
        raise ValueError("n_nodes must be >= 2")
    if n_layers < 2:
        raise ValueError("n_layers must be >= 2")
    r = rng or random.Random()
    n_layers = min(n_layers, n_nodes)

    layers: list[list[int]] = [[] for _ in range(n_layers)]
    for i in range(n_nodes):
        layers[i % n_layers].append(i)

    agent_types = [AGENT_POOL[i % len(AGENT_POOL)] for i in range(n_nodes)]
    durations = {
        i: round(r.uniform(duration_lo, duration_hi), 2) for i in range(n_nodes)
    }

    edges: list[tuple[int, int]] = []
    for li in range(n_layers - 1):
        srcs = layers[li]
        dsts = layers[li + 1]
        for u in srcs:
            linked = False
            for v in dsts:
                if r.random() < edge_prob:
                    edges.append((u, v))
                    linked = True
            if not linked and dsts:
                edges.append((u, r.choice(dsts)))
        for v in dsts:
            if not any(e[1] == v for e in edges) and srcs:
                edges.append((r.choice(srcs), v))

    tasks: dict[str, Task] = {}
    for i in range(n_nodes):
        tid = f"{workflow_id}_n{i}"
        tasks[tid] = Task(
            task_id=tid,
            agent_type=agent_types[i],
            workflow_id=workflow_id,
            expected_duration=durations[i],
            actual_duration=durations[i],
            arrival_time=arrival_time,
        )
    edge_ids = [(f"{workflow_id}_n{u}", f"{workflow_id}_n{v}") for u, v in edges]
    return WorkflowDAG(
        workflow_id=workflow_id,
        tasks=tasks,
        edges=edge_ids,
        arrival_time=arrival_time,
    )


def create_random_dag_family(
    n_dags: int = 24,
    *,
    seed: int = 42,
    n_nodes: int = 8,
    n_layers: int = 4,
    arrival_stagger: float = 0.0,
) -> list[WorkflowDAG]:
    r = random.Random(seed)
    out: list[WorkflowDAG] = []
    for i in range(n_dags):
        out.append(
            create_random_layered_dag(
                workflow_id=f"rand_{i}",
                n_nodes=n_nodes,
                n_layers=n_layers,
                edge_prob=r.uniform(0.4, 0.75),
                arrival_time=i * arrival_stagger,
                rng=random.Random(seed + i * 17),
            )
        )
    return out
