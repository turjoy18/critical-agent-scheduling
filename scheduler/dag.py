"""Workflow DAG and Task representations for multi-agent scheduling."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import networkx as nx

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LLM_PROFILE_CSV = _REPO_ROOT / "llm" / "profiling_summary_llm.csv"
_SIMULATED_PROFILE_CSV = _REPO_ROOT / "simulated" / "profiling_summary.csv"


def load_profile_means(csv_path: Path | str, decimals: int = 2) -> dict[str, float]:
    """Load agent-level mean runtimes from a profiling summary CSV.

    Expects columns ``agent`` and ``avg_runtime_s``. Means are rounded to
    ``decimals`` places so simulator tables stay stable across reloads.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Profiling summary CSV not found: {path}")
    means: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "agent" not in reader.fieldnames:
            raise ValueError(f"CSV missing 'agent' column: {path}")
        if "avg_runtime_s" not in reader.fieldnames:
            raise ValueError(f"CSV missing 'avg_runtime_s' column: {path}")
        for row in reader:
            agent = row["agent"].strip()
            means[agent] = round(float(row["avg_runtime_s"]), decimals)
    if not means:
        raise ValueError(f"No agent means found in {path}")
    return means


# Empirical execution profiles (seconds): loaded from profiling CSVs.
LLM_PROFILE_MEANS: dict[str, float] = load_profile_means(_LLM_PROFILE_CSV)
SIMULATED_PROFILE_MEANS: dict[str, float] = load_profile_means(_SIMULATED_PROFILE_CSV)

RESEARCH_WRITING_EDGES: list[tuple[str, str]] = [
    ("planner", "researcher_a"),
    ("planner", "researcher_b"),
    ("planner", "researcher_c"),
    ("researcher_a", "synthesizer"),
    ("researcher_b", "synthesizer"),
    ("researcher_c", "synthesizer"),
    ("synthesizer", "critic"),
    ("critic", "final_writer"),
]


@dataclass
class Task:
    """An individual agent invocation task within a workflow."""

    task_id: str
    agent_type: str
    workflow_id: str
    expected_duration: float
    actual_duration: float
    arrival_time: float = 0.0
    ready_time: float | None = None
    start_time: float | None = None
    finish_time: float | None = None
    assigned_worker: int | None = None

    @property
    def waiting_time(self) -> float:
        """Time spent in ready queue before execution starts."""
        if self.start_time is None or self.ready_time is None:
            return 0.0
        return max(0.0, self.start_time - self.ready_time)

    @property
    def turnaround_time(self) -> float:
        """Total time from workflow arrival to task completion."""
        if self.finish_time is None:
            return 0.0
        return max(0.0, self.finish_time - self.arrival_time)


@dataclass
class WorkflowDAG:
    """A Directed Acyclic Graph of agent tasks representing a multi-agent workflow."""

    workflow_id: str
    tasks: dict[str, Task]
    edges: list[tuple[str, str]]
    arrival_time: float = 0.0
    predecessors: dict[str, list[str]] = field(default_factory=dict)
    successors: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.predecessors = {node: [] for node in self.tasks}
        self.successors = {node: [] for node in self.tasks}
        for u, v in self.edges:
            if u not in self.tasks or v not in self.tasks:
                raise ValueError(f"Edge ({u}, {v}) references unknown task.")
            self.successors[u].append(v)
            self.predecessors[v].append(u)

    def to_networkx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for task_id, task in self.tasks.items():
            g.add_node(task_id, task=task)
        g.add_edges_from(self.edges)
        return g

    def topological_order(self) -> list[str]:
        g = self.to_networkx()
        return list(nx.topological_sort(g))

    def compute_upward_ranks(self, use_expected: bool = True) -> dict[str, float]:
        """Compute HEFT-style upward rank rank_u for all tasks.

        rank_u(i) = duration(i) + max_{j in succ(i)} rank_u(j)
        """
        ranks: dict[str, float] = {}
        order = list(reversed(self.topological_order()))

        for task_id in order:
            task = self.tasks[task_id]
            dur = task.expected_duration if use_expected else task.actual_duration
            succs = self.successors[task_id]
            if not succs:
                ranks[task_id] = dur
            else:
                ranks[task_id] = dur + max(ranks[succ] for succ in succs)

        return ranks

    def compute_downward_ranks(self, use_expected: bool = True) -> dict[str, float]:
        """Compute downward rank rank_d for all tasks.

        rank_d(i) = max_{j in pred(i)} (rank_d(j) + duration(j))
        """
        ranks: dict[str, float] = {}
        order = self.topological_order()

        for task_id in order:
            preds = self.predecessors[task_id]
            if not preds:
                ranks[task_id] = 0.0
            else:
                ranks[task_id] = max(
                    ranks[p]
                    + (
                        self.tasks[p].expected_duration
                        if use_expected
                        else self.tasks[p].actual_duration
                    )
                    for p in preds
                )

        return ranks

    def compute_critical_path(
        self, use_expected: bool = True
    ) -> tuple[list[str], float]:
        """Compute the longest (critical) path through the workflow."""
        upward = self.compute_upward_ranks(use_expected=use_expected)
        downward = self.compute_downward_ranks(use_expected=use_expected)

        # Total makespan on unconstrained critical path
        path_lengths = {
            t_id: downward[t_id] + upward[t_id] for t_id in self.tasks
        }
        max_len = max(path_lengths.values()) if path_lengths else 0.0

        # Backtrack from node with max upward rank among source nodes
        sources = [t for t in self.tasks if not self.predecessors[t]]
        current = max(sources, key=lambda s: upward[s]) if sources else None

        path = []
        while current is not None:
            path.append(current)
            succs = self.successors[current]
            if not succs:
                break
            current = max(succs, key=lambda s: upward[s])

        return path, max_len

    def compute_static_slacks(
        self, deadline: float | None = None, use_expected: bool = True
    ) -> dict[str, float]:
        """Compute static slack for each task: Slack(i) = Deadline - (rank_d(i) + rank_u(i))."""
        upward = self.compute_upward_ranks(use_expected=use_expected)
        downward = self.compute_downward_ranks(use_expected=use_expected)
        min_makespan = max(downward[t] + upward[t] for t in self.tasks)
        effective_deadline = deadline if deadline is not None else min_makespan

        slacks = {}
        for t in self.tasks:
            total_path_len = downward[t] + upward[t]
            slacks[t] = effective_deadline - total_path_len
        return slacks


def create_research_writing_workflow(
    workflow_id: str = "wf_0",
    profile: Literal["llm", "simulated"] = "llm",
    duration_noise_std: float = 0.0,
    arrival_time: float = 0.0,
    rng: random.Random | None = None,
    custom_durations: dict[str, float] | None = None,
) -> WorkflowDAG:
    """Create a research-writing workflow DAG populated with durations."""
    r = rng if rng is not None else random.Random()
    base_means = LLM_PROFILE_MEANS if profile == "llm" else SIMULATED_PROFILE_MEANS
    if custom_durations:
        base_means = {**base_means, **custom_durations}

    tasks: dict[str, Task] = {}
    for agent in base_means:
        mean_dur = base_means[agent]
        if duration_noise_std > 0.0:
            actual = max(0.05, r.gauss(mean_dur, mean_dur * duration_noise_std))
        else:
            actual = mean_dur

        task_id = f"{workflow_id}_{agent}" if workflow_id else agent
        tasks[task_id] = Task(
            task_id=task_id,
            agent_type=agent,
            workflow_id=workflow_id,
            expected_duration=mean_dur,
            actual_duration=actual,
            arrival_time=arrival_time,
        )

    edges: list[tuple[str, str]] = []
    for u, v in RESEARCH_WRITING_EDGES:
        u_id = f"{workflow_id}_{u}" if workflow_id else u
        v_id = f"{workflow_id}_{v}" if workflow_id else v
        edges.append((u_id, v_id))

    return WorkflowDAG(
        workflow_id=workflow_id,
        tasks=tasks,
        edges=edges,
        arrival_time=arrival_time,
    )
