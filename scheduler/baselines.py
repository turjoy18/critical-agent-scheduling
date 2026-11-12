"""Paper baselines: compute-only, cache-only wrappers, and two-phase CPF→STE.

Two-phase (ablation): ready queue ordered by CPF; cache keep/evict by STE
(steps-to-execution). Couloir replaces both with one utility.
"""

from __future__ import annotations

from typing import Literal

from kvflow.step_graph import AgentStepGraph, steps_to_execution
from scheduler.couloir import CacheAction, CacheView, PrefixKeyFn, agent_type_key
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
    key_fn: PrefixKeyFn | None = None,
) -> CacheAction:
    """STE eviction: prefer evicting agents with largest steps-to-execution."""
    key = key_fn or agent_type_key

    def _type_of(prefix_key: str) -> str:
        return prefix_key.split(":", 1)[-1]

    # Active set = ready/selected only (not residents); residents get STE vs that frontier.
    active = [t.agent_type for t in ready_tasks]
    if selected is not None:
        active = [selected.agent_type, *active]
    ste = ste_from_workflows(workflows, active)
    on_gpu = [a for a, ok in cache.agent_on_gpu.items() if ok]
    ste_by_key: dict[str, float] = {}
    for t in ready_tasks:
        ste_by_key[key(t)] = ste.get(t.agent_type, 1e9)
    if selected is not None:
        ste_by_key[key(selected)] = ste.get(selected.agent_type, 1e9)
    for pk in on_gpu:
        ste_by_key.setdefault(pk, float(ste.get(_type_of(pk), 1e9)))
    keep: list[str] = []
    if selected is not None:
        keep.append(key(selected))
    ranked = sorted(
        {key(t) for t in ready_tasks},
        key=lambda k: (ste_by_key.get(k, 1e9), k),
    )
    for k in ranked:
        if k not in keep:
            keep.append(k)
        if len(keep) >= max(1, prefetch_top_k + 1):
            break
    evict = sorted(
        [a for a in on_gpu if a not in keep],
        key=lambda a: (-ste_by_key.get(a, 0.0), a),
    )
    sel = key(selected) if selected is not None else None
    prefetch = [a for a in keep if a != sel]
    return CacheAction(keep_agents=keep, evict_agents=evict, prefetch_agents=prefetch)


def plan_lru_cache_actions(
    selected: Task | None,
    ready_tasks: list[Task],
    cache: CacheView,
    last_access: dict[str, float],
    current_time: float,
    prefetch_top_k: int = 0,
    key_fn: PrefixKeyFn | None = None,
) -> CacheAction:
    key = key_fn or agent_type_key
    keep: list[str] = []
    if selected is not None:
        keep.append(key(selected))
    on_gpu = [a for a, ok in cache.agent_on_gpu.items() if ok]
    evict = sorted(
        [a for a in on_gpu if a not in keep],
        key=lambda a: (last_access.get(a, -1.0), a),
    )
    return CacheAction(keep_agents=keep, evict_agents=evict, prefetch_agents=[])
