"""Couloir: joint compute + prefix-cache utility for multi-agent DAG scheduling.

Utility for ready task i at time t:

    U(i,t) = w_cp * r_cp(i)
           + w_slack * (1 - r_slack(i))
           + w_reuse * reuse(i)
           - w_evict * evict_cost(i)
           + w_age * age(i,t)

where r_* are min-max normalized over the current ready set (0 if degenerate).
Aging age(i,t) = (t - ready_time(i)) / max(1, max wait in ready set) so starved
tasks eventually dominate compute-only terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from scheduler.dag import Task, WorkflowDAG

# Consensus after batch grid (not a sharp optimum; shape-best often reuse=0.75).
# Aging kept >0 for starvation unit tests.
DEFAULT_WEIGHTS = {
    "cp": 1.0,
    "slack": 0.25,
    "reuse": 1.5,
    "evict": 0.5,
    "age": 0.25,
}

PrefixKeyFn = Callable[[Task], str]


def agent_type_key(task: Task) -> str:
    """Cross-workflow reuse if agent_type strings collide (legacy / optimistic)."""
    return task.agent_type


def workflow_agent_key(task: Task) -> str:
    """No cross-workflow prefix sharing (closer to distinct prompts)."""
    return f"{task.workflow_id}:{task.agent_type}"


@dataclass(frozen=True)
class CouloirWeights:
    cp: float = DEFAULT_WEIGHTS["cp"]
    slack: float = DEFAULT_WEIGHTS["slack"]
    reuse: float = DEFAULT_WEIGHTS["reuse"]
    evict: float = DEFAULT_WEIGHTS["evict"]
    age: float = DEFAULT_WEIGHTS["age"]


@dataclass
class CacheView:
    """Minimal cache state Couloir needs (agent-keyed prefix residency)."""

    capacity_tokens: int
    gpu_tokens: int
    agent_on_gpu: dict[str, bool] = field(default_factory=dict)
    agent_prefix_tokens: dict[str, int] = field(default_factory=dict)
    agent_ste: dict[str, float] = field(default_factory=dict)

    def reuse_benefit(self, agent: str) -> float:
        if self.agent_on_gpu.get(agent, False):
            return 1.0
        return 0.0

    def eviction_cost(self, agent: str) -> float:
        """Relative cost of bringing ``agent`` onto GPU under capacity pressure."""
        need = self.agent_prefix_tokens.get(agent, 0)
        if need <= 0:
            return 0.0
        free = max(0, self.capacity_tokens - self.gpu_tokens)
        if self.agent_on_gpu.get(agent, False) or need <= free:
            return 0.0
        # Pressure in [0, 1]: how much of capacity must be freed.
        return min(1.0, (need - free) / max(1, self.capacity_tokens))


@dataclass(frozen=True)
class CouloirScoreBreakdown:
    task_id: str
    agent_type: str
    utility: float
    cp: float
    slack: float
    reuse: float
    evict: float
    age: float
    raw_cp: float
    raw_slack: float
    raw_age_s: float


@dataclass
class CacheAction:
    """Evict / keep / prefetch decisions emitted with a Couloir dispatch."""

    keep_agents: list[str] = field(default_factory=list)
    evict_agents: list[str] = field(default_factory=list)
    prefetch_agents: list[str] = field(default_factory=list)


@dataclass
class CouloirDecision:
    ordered_tasks: list[Task]
    scores: list[CouloirScoreBreakdown]
    cache_action: CacheAction
    selected_task_id: str | None = None


def _minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def remaining_critical_path(
    task: Task,
    workflows: dict[str, WorkflowDAG],
    ranks_cache: dict[str, dict[str, float]] | None = None,
) -> float:
    wf = workflows[task.workflow_id]
    if ranks_cache is not None:
        if task.workflow_id not in ranks_cache:
            ranks_cache[task.workflow_id] = wf.compute_upward_ranks(use_expected=True)
        return ranks_cache[task.workflow_id].get(task.task_id, task.expected_duration)
    return wf.compute_upward_ranks(use_expected=True).get(
        task.task_id, task.expected_duration
    )


def task_slack(
    task: Task,
    current_time: float,
    workflows: dict[str, WorkflowDAG],
    ranks_cache: dict[str, dict[str, float]] | None = None,
    deadlines_cache: dict[str, float] | None = None,
    deadline_multiplier: float = 1.0,
) -> float:
    wf_id = task.workflow_id
    wf = workflows[wf_id]
    if ranks_cache is None:
        ranks_cache = {}
    if deadlines_cache is None:
        deadlines_cache = {}
    if wf_id not in ranks_cache:
        ranks_cache[wf_id] = wf.compute_upward_ranks(use_expected=True)
    if wf_id not in deadlines_cache:
        _, min_makespan = wf.compute_critical_path(use_expected=True)
        deadlines_cache[wf_id] = task.arrival_time + min_makespan * deadline_multiplier
    upward = ranks_cache[wf_id].get(task.task_id, task.expected_duration)
    return deadlines_cache[wf_id] - (current_time + upward)


def wait_age_seconds(task: Task, current_time: float) -> float:
    ready = task.ready_time if task.ready_time is not None else task.arrival_time
    return max(0.0, current_time - ready)


def score_ready_tasks(
    ready_tasks: Sequence[Task],
    current_time: float,
    workflows: dict[str, WorkflowDAG],
    cache: CacheView | None = None,
    weights: CouloirWeights | None = None,
    enable_aging: bool = True,
    key_fn: PrefixKeyFn | None = None,
) -> list[CouloirScoreBreakdown]:
    """Score every ready task. Deterministic for fixed inputs."""
    if not ready_tasks:
        return []
    w = weights or CouloirWeights()
    cache = cache or CacheView(capacity_tokens=1, gpu_tokens=0)
    key = key_fn or agent_type_key
    ranks: dict[str, dict[str, float]] = {}
    deadlines: dict[str, float] = {}

    raw_cp = [remaining_critical_path(t, workflows, ranks) for t in ready_tasks]
    raw_slack = [
        task_slack(t, current_time, workflows, ranks, deadlines) for t in ready_tasks
    ]
    raw_age = [wait_age_seconds(t, current_time) for t in ready_tasks]
    raw_reuse = [cache.reuse_benefit(key(t)) for t in ready_tasks]
    raw_evict = [cache.eviction_cost(key(t)) for t in ready_tasks]

    n_cp = _minmax(raw_cp)
    # Lower slack → higher urgency; normalize then invert in the utility mix.
    n_slack = _minmax(raw_slack)
    n_age = _minmax(raw_age) if enable_aging else [0.0] * len(ready_tasks)

    out: list[CouloirScoreBreakdown] = []
    for i, t in enumerate(ready_tasks):
        utility = (
            w.cp * n_cp[i]
            + w.slack * (1.0 - n_slack[i])
            + w.reuse * raw_reuse[i]
            - w.evict * raw_evict[i]
            + w.age * n_age[i]
        )
        out.append(
            CouloirScoreBreakdown(
                task_id=t.task_id,
                agent_type=t.agent_type,
                utility=utility,
                cp=n_cp[i],
                slack=n_slack[i],
                reuse=raw_reuse[i],
                evict=raw_evict[i],
                age=n_age[i],
                raw_cp=raw_cp[i],
                raw_slack=raw_slack[i],
                raw_age_s=raw_age[i],
            )
        )
    return out


def order_by_utility(
    ready_tasks: Sequence[Task],
    scores: Sequence[CouloirScoreBreakdown],
) -> list[Task]:
    by_id = {t.task_id: t for t in ready_tasks}
    ranked = sorted(
        scores,
        key=lambda s: (-s.utility, s.raw_slack, s.task_id),
    )
    return [by_id[s.task_id] for s in ranked]


def plan_cache_actions(
    selected: Task | None,
    ready_tasks: Sequence[Task],
    scores: Sequence[CouloirScoreBreakdown],
    cache: CacheView,
    prefetch_top_k: int = 1,
    key_fn: PrefixKeyFn | None = None,
) -> CacheAction:
    """Keep selected + high-utility ready agents; evict lowest-utility residents."""
    key = key_fn or agent_type_key
    by_id = {t.task_id: t for t in ready_tasks}
    if selected is not None:
        by_id[selected.task_id] = selected
    score_by_key: dict[str, CouloirScoreBreakdown] = {}
    for s in scores:
        t = by_id.get(s.task_id)
        if t is not None:
            score_by_key[key(t)] = s
    keep: list[str] = []
    if selected is not None:
        keep.append(key(selected))
    ranked_keys = sorted(
        {key(t) for t in ready_tasks},
        key=lambda k: (
            -(score_by_key[k].utility if k in score_by_key else 0.0),
            k,
        ),
    )
    for k in ranked_keys:
        if k not in keep:
            keep.append(k)
        if len(keep) >= max(1, prefetch_top_k + 1):
            break

    on_gpu = [a for a, ok in cache.agent_on_gpu.items() if ok]
    evict = sorted(
        [a for a in on_gpu if a not in keep],
        key=lambda a: (
            score_by_key[a].utility if a in score_by_key else -1.0,
            -cache.agent_ste.get(a, 0.0),
            a,
        ),
    )
    sel_key = key(selected) if selected is not None else None
    prefetch = [a for a in keep if a != sel_key]
    return CacheAction(keep_agents=keep, evict_agents=evict, prefetch_agents=prefetch)


def decide(
    ready_tasks: Sequence[Task],
    current_time: float,
    workflows: dict[str, WorkflowDAG],
    cache: CacheView | None = None,
    weights: CouloirWeights | None = None,
    enable_aging: bool = True,
    prefetch_top_k: int = 1,
    key_fn: PrefixKeyFn | None = None,
) -> CouloirDecision:
    key = key_fn or agent_type_key
    cache = cache or CacheView(capacity_tokens=1, gpu_tokens=0)
    scores = score_ready_tasks(
        ready_tasks,
        current_time,
        workflows,
        cache,
        weights,
        enable_aging,
        key_fn=key,
    )
    ordered = order_by_utility(ready_tasks, scores)
    selected = ordered[0] if ordered else None
    action = plan_cache_actions(
        selected,
        ready_tasks,
        scores,
        cache,
        prefetch_top_k=prefetch_top_k,
        key_fn=key,
    )
    return CouloirDecision(
        ordered_tasks=ordered,
        scores=scores,
        cache_action=action,
        selected_task_id=selected.task_id if selected else None,
    )


def couloir_policy(
    ready_tasks: list[Task],
    current_time: float,
    workflows: dict[str, WorkflowDAG],
) -> list[Task]:
    """Compute-only Couloir ordering (no live cache view). Compatible with SimulationEngine."""
    return decide(ready_tasks, current_time, workflows).ordered_tasks


def eviction_priority_agents(
    agents: Iterable[str],
    scores_by_agent: dict[str, float],
    ste: dict[str, float] | None = None,
) -> list[str]:
    """Higher list position = more evictable (low utility, high STE)."""
    ste = ste or {}
    return sorted(
        agents,
        key=lambda a: (scores_by_agent.get(a, 0.0), -ste.get(a, 0.0), a),
    )
