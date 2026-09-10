"""Scheduler package for multi-agent LLM workflows."""

from __future__ import annotations

from scheduler.dag import (
    LLM_PROFILE_MEANS,
    SIMULATED_PROFILE_MEANS,
    Task,
    WorkflowDAG,
    create_research_writing_workflow,
    load_profile_means,
)
from scheduler.engine import SimulationEngine, SimulationResult, WorkerPool
from scheduler.metrics import MetricsCollector, SchedulingMetrics
from scheduler.policies import (
    POLICIES,
    critical_path_first_policy,
    fifo_policy,
    get_policy,
    heft_policy,
    least_slack_first_policy,
)

__all__ = [
    "Task",
    "WorkflowDAG",
    "create_research_writing_workflow",
    "LLM_PROFILE_MEANS",
    "SIMULATED_PROFILE_MEANS",
    "load_profile_means",
    "SimulationEngine",
    "SimulationResult",
    "WorkerPool",
    "MetricsCollector",
    "SchedulingMetrics",
    "POLICIES",
    "get_policy",
    "fifo_policy",
    "critical_path_first_policy",
    "least_slack_first_policy",
    "heft_policy",
]
