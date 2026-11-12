"""Scheduling policies for multi-agent DAG execution: FIFO, CPF, Slack, HEFT, Couloir."""

from __future__ import annotations

from typing import Callable

from scheduler.couloir import couloir_policy
from scheduler.dag import Task, WorkflowDAG

PolicyFunction = Callable[[list[Task], float, dict[str, WorkflowDAG]], list[Task]]


def fifo_policy(
    ready_tasks: list[Task],
    current_time: float,
    workflows: dict[str, WorkflowDAG],
) -> list[Task]:
    """FIFO / First-Come-First-Served.

    Sorts by ready_time ascending, then task_id for stability.
    """
    return sorted(
        ready_tasks,
        key=lambda t: (
            t.ready_time if t.ready_time is not None else t.arrival_time,
            t.arrival_time,
            t.task_id,
        ),
    )


def critical_path_first_policy(
    ready_tasks: list[Task],
    current_time: float,
    workflows: dict[str, WorkflowDAG],
) -> list[Task]:
    """Critical-Path-First (CPF).

    Prioritizes tasks with the largest remaining path length (upward rank)
    to the workflow sink. Tasks on the critical path finish earlier, unblocking downstream joins.
    """
    # Cache upward ranks per workflow
    upward_ranks_cache: dict[str, dict[str, float]] = {}

    def get_upward_rank(task: Task) -> float:
        wf_id = task.workflow_id
        if wf_id not in upward_ranks_cache:
            wf = workflows[wf_id]
            upward_ranks_cache[wf_id] = wf.compute_upward_ranks(use_expected=True)
        return upward_ranks_cache[wf_id].get(task.task_id, task.expected_duration)

    return sorted(
        ready_tasks,
        key=lambda t: (-get_upward_rank(t), t.ready_time or 0.0, t.task_id),
    )


def least_slack_first_policy(
    ready_tasks: list[Task],
    current_time: float,
    workflows: dict[str, WorkflowDAG],
    target_deadline_multiplier: float = 1.0,
) -> list[Task]:
    """Least-Slack-First (LSF) / Slack-based Prioritization.

    Slack(i, t) = Deadline - (current_time + remaining_critical_path(i))
    Tasks with minimum slack are scheduled first to prevent makespan violations.
    """
    upward_ranks_cache: dict[str, dict[str, float]] = {}
    deadlines_cache: dict[str, float] = {}

    def get_task_slack(task: Task) -> float:
        wf_id = task.workflow_id
        if wf_id not in upward_ranks_cache:
            wf = workflows[wf_id]
            upward_ranks_cache[wf_id] = wf.compute_upward_ranks(use_expected=True)
            _, min_makespan = wf.compute_critical_path(use_expected=True)
            deadlines_cache[wf_id] = (
                task.arrival_time + min_makespan * target_deadline_multiplier
            )

        upward_rank = upward_ranks_cache[wf_id].get(
            task.task_id, task.expected_duration
        )
        deadline = deadlines_cache[wf_id]
        # Slack = Deadline - (Current Time + Remaining Path Length)
        slack = deadline - (current_time + upward_rank)
        return slack

    return sorted(
        ready_tasks,
        key=lambda t: (get_task_slack(t), -t.expected_duration, t.task_id),
    )


def heft_policy(
    ready_tasks: list[Task],
    current_time: float,
    workflows: dict[str, WorkflowDAG],
) -> list[Task]:
    """Simplified HEFT (Heterogeneous Earliest Finish Time) Upward Rank Policy.

    Computes upward rank rank_u(i) = w_i + max_{j in succ(i)} rank_u(j).
    In a worker pool setting, tasks with highest rank_u are prioritized for the earliest available slot.
    Tie-breaking favors earlier ready time, then arrival time.
    """
    upward_ranks_cache: dict[str, dict[str, float]] = {}

    def get_heft_rank(task: Task) -> float:
        wf_id = task.workflow_id
        if wf_id not in upward_ranks_cache:
            wf = workflows[wf_id]
            upward_ranks_cache[wf_id] = wf.compute_upward_ranks(use_expected=True)
        return upward_ranks_cache[wf_id].get(task.task_id, task.expected_duration)

    return sorted(
        ready_tasks,
        key=lambda t: (
            -get_heft_rank(t),
            t.ready_time if t.ready_time is not None else 0.0,
            t.arrival_time,
            t.task_id,
        ),
    )


POLICIES: dict[str, PolicyFunction] = {
    "fifo": fifo_policy,
    "cpf": critical_path_first_policy,
    "slack": least_slack_first_policy,
    "heft": heft_policy,
    "couloir": couloir_policy,
}


def get_policy(name: str) -> PolicyFunction:
    """Retrieve scheduling policy function by name."""
    name_clean = name.strip().lower()
    if name_clean not in POLICIES:
        raise ValueError(
            f"Unknown policy '{name}'. Available: {list(POLICIES.keys())}"
        )
    return POLICIES[name_clean]
