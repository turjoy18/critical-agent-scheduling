"""Paper baselines: compute-only, cache-only wrappers, and two-phase CPF→STE.

Two-phase (ablation): ready queue ordered by CPF; cache keep/evict by STE
(steps-to-execution). Couloir replaces both with one utility.
"""

from __future__ import annotations

from typing import Literal

from kvflow.step_graph import AgentStepGraph, steps_to_execution
from scheduler.couloir import CacheAction, CacheView
from scheduler.dag import Task, WorkflowDAG

CacheMode = Literal["none", "lru", "ste", "couloir"]


def ste_from_workflows(
    workflows: dict[str, WorkflowDAG],
    active_agents: list[str],
) -> dict[str, float]:
    """Aggregate STE across workflows (min STE if agent appears in several)."""
    combined: dict[str, float] = {}
    for wf in workflows.values():
        agents = sorted({t.agent_type for t in wf.tasks.values()})
        # Edges in agent-type space (drop workflow_id prefix semantics).
        type_edges: list[tuple[str, str]] = []
        for u, v in wf.edges:
            ua = wf.tasks[u].agent_type
            va = wf.tasks[v].agent_type
            if (ua, va) not in type_edges:
                type_edges.append((ua, va))
        graph = AgentStepGraph(nodes=agents, edges=type_edges)
        active = [a for a in active_agents if a in agents]
        if not active:
            continue
        ste = steps_to_execution(graph, active)
        for a, s in ste.items():
            prev = combined.get(a)
            combined[a] = float(s) if prev is None else min(prev, float(s))
    return combined


def plan_ste_cache_actions(
    selected: Task | None,
    ready_tasks: list[Task],
    workflows: dict[str, WorkflowDAG],
    cache: CacheView,
    prefetch_top_k: int = 1,
) -> CacheAction:
    """STE eviction: prefer evicting agents with largest steps-to-execution."""
    active = [t.agent_type for t in ready_tasks]
    if selected is not None:
        active = [selected.agent_type, *active]
    ste = ste_from_workflows(workflows, active)
    keep: list[str] = []
    if selected is not None:
        keep.append(selected.agent_type)
    # Keep soonest-to-execute ready agents (low STE).
    ranked = sorted(
        {t.agent_type for t in ready_tasks},
        key=lambda a: (ste.get(a, 1e9), a),
    )
    for a in ranked:
        if a not in keep:
            keep.append(a)
        if len(keep) >= max(1, prefetch_top_k + 1):
            break
    on_gpu = [a for a, ok in cache.agent_on_gpu.items() if ok]
    evict = sorted(
        [a for a in on_gpu if a not in keep],
        key=lambda a: (-ste.get(a, 0.0), a),  # high STE first
    )
    prefetch = [a for a in keep if selected is None or a != selected.agent_type]
    return CacheAction(keep_agents=keep, evict_agents=evict, prefetch_agents=prefetch)


def plan_lru_cache_actions(
    selected: Task | None,
    ready_tasks: list[Task],
    cache: CacheView,
    last_access: dict[str, float],
    current_time: float,
    prefetch_top_k: int = 0,
) -> CacheAction:
    keep: list[str] = []
    if selected is not None:
        keep.append(selected.agent_type)
    on_gpu = [a for a, ok in cache.agent_on_gpu.items() if ok]
    evict = sorted(
        [a for a in on_gpu if a not in keep],
        key=lambda a: (last_access.get(a, -1.0), a),  # oldest first
    )
    return CacheAction(keep_agents=keep, evict_agents=evict, prefetch_agents=[])
