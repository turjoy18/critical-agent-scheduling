"""Metrics collection and reporting for multi-agent scheduling evaluations.

Unified schema (sim + engine): makespan, JCT p50/p99, cache hit rate, recompute tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from scheduler.dag import Task, WorkflowDAG

# Canonical result keys for paper tables / engine parity.
METRICS_SCHEMA_VERSION = 1
REQUIRED_RESULT_KEYS = (
    "schema_version",
    "policy",
    "num_workers",
    "makespan",
    "jct_p50",
    "jct_p99",
    "cache_hit_rate",
    "recompute_tokens",
)


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


@dataclass
class SchedulingMetrics:
    """Summary metrics for a scheduled batch or single workflow execution."""

    policy_name: str
    num_workers: int
    makespan: float
    unconstrained_cpath_len: float
    critical_path_delay: float
    avg_waiting_time: float
    max_waiting_time: float
    avg_turnaround_time: float
    worker_utilization: float
    total_compute_time: float
    num_tasks: int
    num_workflows: int
    jct_p50: float = 0.0
    jct_p99: float = 0.0
    cache_hit_rate: float | None = None
    recompute_tokens: int | None = None
    cache_evictions: int | None = None
    per_agent_waiting: dict[str, float] = field(default_factory=dict)
    per_agent_turnaround: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | int | str | None]:
        base: dict[str, float | int | str | None] = {
            "schema_version": METRICS_SCHEMA_VERSION,
            "policy": self.policy_name,
            "num_workers": self.num_workers,
            "makespan": round(self.makespan, 4),
            "unconstrained_cpath": round(self.unconstrained_cpath_len, 4),
            "cpath_delay": round(self.critical_path_delay, 4),
            "avg_waiting_time": round(self.avg_waiting_time, 4),
            "max_waiting_time": round(self.max_waiting_time, 4),
            "avg_turnaround_time": round(self.avg_turnaround_time, 4),
            "jct_p50": round(self.jct_p50, 4),
            "jct_p99": round(self.jct_p99, 4),
            "worker_utilization": round(self.worker_utilization, 4),
            "total_compute_time": round(self.total_compute_time, 4),
            "num_tasks": self.num_tasks,
            "num_workflows": self.num_workflows,
            "cache_hit_rate": (
                round(self.cache_hit_rate, 4) if self.cache_hit_rate is not None else None
            ),
            "recompute_tokens": self.recompute_tokens,
            "cache_evictions": self.cache_evictions,
        }
        for agent, wait in self.per_agent_waiting.items():
            base[f"wait_{agent}"] = round(wait, 4)
        return base


class MetricsCollector:
    """Computes scheduling metrics from completed workflows and worker execution records."""

    @staticmethod
    def calculate(
        policy_name: str,
        num_workers: int,
        workflows: Sequence[WorkflowDAG],
        sim_start_time: float = 0.0,
        cache_stats: dict[str, Any] | None = None,
    ) -> SchedulingMetrics:
        all_tasks: list[Task] = []
        for wf in workflows:
            all_tasks.extend(wf.tasks.values())

        if not all_tasks:
            raise ValueError("Cannot calculate metrics on empty task list.")

        finish_times = [t.finish_time for t in all_tasks if t.finish_time is not None]
        if len(finish_times) != len(all_tasks):
            raise RuntimeError("Not all tasks have finished execution.")

        makespan = max(finish_times) - sim_start_time
        total_compute = sum(t.actual_duration for t in all_tasks)

        # Theoretical ideal critical path (single workflow or max across independent workflows)
        ideal_cpaths = [
            wf.compute_critical_path(use_expected=False)[1] for wf in workflows
        ]
        max_ideal_cpath = max(ideal_cpaths) if ideal_cpaths else 0.0
        cpath_delay = max(0.0, makespan - max_ideal_cpath)

        waiting_times = [t.waiting_time for t in all_tasks]
        turnaround_times = [t.turnaround_time for t in all_tasks]

        # Workflow-level JCT: last task finish − arrival
        jcts: list[float] = []
        for wf in workflows:
            finishes = [t.finish_time for t in wf.tasks.values() if t.finish_time is not None]
            if finishes:
                jcts.append(max(finishes) - wf.arrival_time)
        jcts_sorted = sorted(jcts)

        utilization = total_compute / (num_workers * makespan) if makespan > 0 else 0.0
        utilization = min(1.0, max(0.0, utilization))

        # Breakdown by agent type
        agent_waits: dict[str, list[float]] = {}
        agent_turns: dict[str, list[float]] = {}
        for t in all_tasks:
            agent_waits.setdefault(t.agent_type, []).append(t.waiting_time)
            agent_turns.setdefault(t.agent_type, []).append(t.turnaround_time)

        per_agent_waiting = {
            agent: sum(waits) / len(waits) for agent, waits in agent_waits.items()
        }
        per_agent_turnaround = {
            agent: sum(turns) / len(turns) for agent, turns in agent_turns.items()
        }

        hit_rate = None
        recompute = None
        evictions = None
        if cache_stats:
            hit_rate = float(cache_stats.get("hit_rate", 0.0))
            recompute = int(cache_stats.get("recompute_tokens", 0))
            evictions = int(cache_stats.get("evictions", 0))

        return SchedulingMetrics(
            policy_name=policy_name,
            num_workers=num_workers,
            makespan=makespan,
            unconstrained_cpath_len=max_ideal_cpath,
            critical_path_delay=cpath_delay,
            avg_waiting_time=sum(waiting_times) / len(waiting_times),
            max_waiting_time=max(waiting_times),
            avg_turnaround_time=sum(turnaround_times) / len(turnaround_times),
            worker_utilization=utilization,
            total_compute_time=total_compute,
            num_tasks=len(all_tasks),
            num_workflows=len(workflows),
            jct_p50=_percentile(jcts_sorted, 50),
            jct_p99=_percentile(jcts_sorted, 99),
            cache_hit_rate=hit_rate,
            recompute_tokens=recompute,
            cache_evictions=evictions,
            per_agent_waiting=per_agent_waiting,
            per_agent_turnaround=per_agent_turnaround,
        )

    @staticmethod
    def to_dataframe(metrics_list: list[SchedulingMetrics]) -> pd.DataFrame:
        """Convert a list of SchedulingMetrics to a formatted DataFrame."""
        return pd.DataFrame([m.to_dict() for m in metrics_list])
