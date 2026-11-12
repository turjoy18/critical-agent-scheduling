"""Named workflow DAG shapes for Couloir multi-DAG evaluation."""

from __future__ import annotations

import random
from typing import Literal

from scheduler.dag import (
    LLM_PROFILE_MEANS,
    RESEARCH_WRITING_EDGES,
    SIMULATED_PROFILE_MEANS,
    Task,
    WorkflowDAG,
    create_research_writing_workflow,
)

DagName = Literal["writing", "chain", "fanout"]

# Deep chain: long spine, almost no parallelism.
CHAIN_AGENTS = [
    "stage_0",
    "stage_1",
    "stage_2",
    "stage_3",
    "stage_4",
    "stage_5",
    "stage_6",
]
CHAIN_DURATIONS = {
    "stage_0": 8.0,
    "stage_1": 12.0,
    "stage_2": 10.0,
    "stage_3": 25.0,  # critical bulge
    "stage_4": 9.0,
    "stage_5": 14.0,
    "stage_6": 11.0,
}
CHAIN_EDGES = [(CHAIN_AGENTS[i], CHAIN_AGENTS[i + 1]) for i in range(len(CHAIN_AGENTS) - 1)]

# Wide fan-out: many parallel peers after a short planner, then join.
FANOUT_PARALLEL = [f"worker_{i}" for i in range(6)]
FANOUT_DURATIONS = {
    "planner": 5.0,
    **{f"worker_{i}": float(8 + 4 * i) for i in range(6)},  # worker_5 longest
    "joiner": 7.0,
}
FANOUT_EDGES = (
    [("planner", w) for w in FANOUT_PARALLEL]
    + [(w, "joiner") for w in FANOUT_PARALLEL]
)


def _build_dag(
    workflow_id: str,
    agents: list[str],
    edges_raw: list[tuple[str, str]],
    durations: dict[str, float],
    arrival_time: float,
    duration_noise_std: float,
    rng: random.Random,
) -> WorkflowDAG:
    tasks: dict[str, Task] = {}
    for agent in agents:
        mean = durations[agent]
        actual = (
            max(0.05, rng.gauss(mean, mean * duration_noise_std))
            if duration_noise_std > 0.0
            else mean
        )
        tid = f"{workflow_id}_{agent}"
        tasks[tid] = Task(
            task_id=tid,
            agent_type=agent,
            workflow_id=workflow_id,
            expected_duration=mean,
            actual_duration=actual,
            arrival_time=arrival_time,
        )
    edges = [
        (f"{workflow_id}_{u}", f"{workflow_id}_{v}") for u, v in edges_raw
    ]
    return WorkflowDAG(
        workflow_id=workflow_id,
        tasks=tasks,
        edges=edges,
        arrival_time=arrival_time,
    )


def create_chain_workflow(
    workflow_id: str = "wf_chain",
    duration_noise_std: float = 0.0,
    arrival_time: float = 0.0,
    rng: random.Random | None = None,
) -> WorkflowDAG:
    r = rng or random.Random()
    return _build_dag(
        workflow_id,
        CHAIN_AGENTS,
        CHAIN_EDGES,
        CHAIN_DURATIONS,
        arrival_time,
        duration_noise_std,
        r,
    )


def create_fanout_workflow(
    workflow_id: str = "wf_fanout",
    duration_noise_std: float = 0.0,
    arrival_time: float = 0.0,
    rng: random.Random | None = None,
) -> WorkflowDAG:
    r = rng or random.Random()
    agents = ["planner", *FANOUT_PARALLEL, "joiner"]
    return _build_dag(
        workflow_id,
        agents,
        FANOUT_EDGES,
        FANOUT_DURATIONS,
        arrival_time,
        duration_noise_std,
        r,
    )


def create_workflow(
    dag: DagName,
    workflow_id: str = "wf0",
    profile: Literal["llm", "simulated"] = "llm",
    duration_noise_std: float = 0.0,
    arrival_time: float = 0.0,
    rng: random.Random | None = None,
) -> WorkflowDAG:
    if dag == "writing":
        return create_research_writing_workflow(
            workflow_id=workflow_id,
            profile=profile,
            duration_noise_std=duration_noise_std,
            arrival_time=arrival_time,
            rng=rng,
        )
    if dag == "chain":
        return create_chain_workflow(
            workflow_id=workflow_id,
            duration_noise_std=duration_noise_std,
            arrival_time=arrival_time,
            rng=rng,
        )
    if dag == "fanout":
        return create_fanout_workflow(
            workflow_id=workflow_id,
            duration_noise_std=duration_noise_std,
            arrival_time=arrival_time,
            rng=rng,
        )
    raise ValueError(f"Unknown dag '{dag}'. Expected writing|chain|fanout")


def dag_agent_means(dag: DagName, profile: Literal["llm", "simulated"] = "llm") -> dict[str, float]:
    if dag == "writing":
        return dict(LLM_PROFILE_MEANS if profile == "llm" else SIMULATED_PROFILE_MEANS)
    if dag == "chain":
        return dict(CHAIN_DURATIONS)
    if dag == "fanout":
        return dict(FANOUT_DURATIONS)
    raise ValueError(dag)


def writing_edges() -> list[tuple[str, str]]:
    return list(RESEARCH_WRITING_EDGES)
