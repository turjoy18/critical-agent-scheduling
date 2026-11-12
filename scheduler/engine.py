"""Discrete-event simulation engine for multi-agent DAG scheduling across worker pools."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Sequence

from scheduler.baselines import plan_lru_cache_actions, plan_ste_cache_actions
from scheduler.couloir import (
    CacheAction,
    CacheView,
    CouloirDecision,
    CouloirWeights,
    decide,
)
from scheduler.dag import Task, WorkflowDAG
from scheduler.metrics import MetricsCollector, SchedulingMetrics


class EventType(IntEnum):
    WORKFLOW_ARRIVAL = 1
    TASK_COMPLETE = 2


@dataclass(order=True)
class SimEvent:
    time: float
    priority: int
    event_type: EventType = field(compare=False)
    payload: Any = field(compare=False)


@dataclass
class Worker:
    worker_id: int
    busy_until: float = 0.0
    current_task: Task | None = None
    total_busy_time: float = 0.0


class WorkerPool:
    """Manages a fixed pool of K worker slots (representing compute threads or LLM API slots)."""

    def __init__(self, num_workers: int) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        self.num_workers = num_workers
        self.workers = [Worker(worker_id=i) for i in range(num_workers)]

    @property
    def idle_workers(self) -> list[Worker]:
        return [w for w in self.workers if w.current_task is None]

    def assign_task(
        self,
        worker: Worker,
        task: Task,
        current_time: float,
        duration: float | None = None,
    ) -> float:
        """Assign task to worker, return finish timestamp.

        ``duration`` overrides ``task.actual_duration`` when cache hit/miss
        adjusts prefill cost; the task field is updated to the effective value.
        """
        run_s = task.actual_duration if duration is None else duration
        task.actual_duration = run_s
        worker.current_task = task
        task.assigned_worker = worker.worker_id
        task.start_time = current_time
        task.finish_time = current_time + run_s
        worker.busy_until = task.finish_time
        worker.total_busy_time += run_s
        return task.finish_time

    def release_worker(self, worker_id: int) -> Worker:
        worker = self.workers[worker_id]
        worker.current_task = None
        return worker


@dataclass
class SimulationResult:
    """Detailed result of a discrete-event simulation run."""

    policy_name: str
    num_workers: int
    workflows: list[WorkflowDAG]
    metrics: SchedulingMetrics
    event_log: list[dict[str, Any]] = field(default_factory=list)
    cache_capacity: int | None = None


# Policy sort function signature: (ready_queue: list[Task], current_time: float, workflows_map: dict[str, WorkflowDAG]) -> list[Task]
PolicyFunction = Callable[[list[Task], float, dict[str, WorkflowDAG]], list[Task]]


# Prefill throughput for miss→hit duration coupling (tok/s). Lower than
# kvflow RECOMPUTE so conceptual prefixes produce visible makespan deltas.
PREFILL_TOKENS_PER_SEC = 1024.0


class JointCacheState:
    """Lightweight token-budget cache tied to agent types for Couloir joint mode."""

    def __init__(
        self,
        capacity_tokens: int,
        prefix_tokens: int = 20,
        agents: Sequence[str] | None = None,
        prefill_tokens_per_sec: float = PREFILL_TOKENS_PER_SEC,
    ) -> None:
        if capacity_tokens < 1:
            raise ValueError("capacity_tokens must be >= 1")
        self.capacity_tokens = capacity_tokens
        self.prefix_tokens = prefix_tokens
        self.prefill_tokens_per_sec = prefill_tokens_per_sec
        self.agent_on_gpu: dict[str, bool] = {a: False for a in (agents or [])}
        self.hits = 0
        self.misses = 0
        self.recompute_tokens = 0
        self.evictions = 0

    def prefill_latency_s(self) -> float:
        return self.prefix_tokens / self.prefill_tokens_per_sec

    def effective_duration(self, base_duration: float, hit: bool) -> float:
        """Profiled duration assumes cold prefill; a hit skips that cost."""
        save = self.prefill_latency_s() if hit else 0.0
        return max(0.05, base_duration - save)

    def view(self) -> CacheView:
        gpu = sum(
            self.prefix_tokens for a, on in self.agent_on_gpu.items() if on
        )
        return CacheView(
            capacity_tokens=self.capacity_tokens,
            gpu_tokens=gpu,
            agent_on_gpu=dict(self.agent_on_gpu),
            agent_prefix_tokens={
                a: self.prefix_tokens for a in self.agent_on_gpu
            },
        )

    def apply_action(self, action: CacheAction, selected_agent: str) -> dict[str, Any]:
        for a in action.evict_agents:
            if self.agent_on_gpu.get(a):
                self.agent_on_gpu[a] = False
                self.evictions += 1
        hit = bool(self.agent_on_gpu.get(selected_agent))
        if hit:
            self.hits += 1
        else:
            self.misses += 1
            self.recompute_tokens += self.prefix_tokens
            while (
                sum(self.prefix_tokens for a, on in self.agent_on_gpu.items() if on)
                + self.prefix_tokens
                > self.capacity_tokens
            ):
                victims = [
                    a for a, on in self.agent_on_gpu.items() if on and a != selected_agent
                ]
                if not victims:
                    break
                victim = victims[0]
                self.agent_on_gpu[victim] = False
                self.evictions += 1
            self.agent_on_gpu[selected_agent] = True
        for a in action.prefetch_agents:
            if a == selected_agent:
                continue
            free = self.capacity_tokens - sum(
                self.prefix_tokens for x, on in self.agent_on_gpu.items() if on
            )
            if free >= self.prefix_tokens:
                self.agent_on_gpu[a] = True
        return {
            "cache_hit": hit,
            "keep": list(action.keep_agents),
            "evict": list(action.evict_agents),
            "prefetch": list(action.prefetch_agents),
        }

    def apply_decision(self, decision: CouloirDecision, selected_agent: str) -> dict[str, Any]:
        return self.apply_action(decision.cache_action, selected_agent)


class SimulationEngine:
    """Discrete-event simulator for multi-agent DAG task execution under worker contention."""

    def __init__(
        self,
        num_workers: int,
        policy_func: PolicyFunction,
        policy_name: str = "custom",
        cache_capacity: int | None = None,
        cache_prefix_tokens: int = 20,
        joint_couloir: bool = False,
        cache_mode: str | None = None,
        enable_aging: bool = True,
        weights: CouloirWeights | None = None,
    ) -> None:
        self.num_workers = num_workers
        self.policy_func = policy_func
        self.policy_name = policy_name
        self.cache_capacity = cache_capacity
        self.cache_prefix_tokens = cache_prefix_tokens
        self.enable_aging = enable_aging
        self.weights = weights
        name = policy_name.strip().lower()
        self.joint_couloir = joint_couloir or name == "couloir"
        # cache_mode: none | lru | ste | couloir (default: couloir if joint, else none)
        if cache_mode is not None:
            self.cache_mode = cache_mode
        elif self.joint_couloir:
            self.cache_mode = "couloir"
        elif name in ("two_phase", "two_phase_cpf_ste", "cpf_ste"):
            self.cache_mode = "ste"
        else:
            self.cache_mode = "none"

    def run(
        self,
        workflows: Sequence[WorkflowDAG],
        sim_start_time: float = 0.0,
    ) -> SimulationResult:
        """Execute the discrete-event simulation on the supplied workflows."""
        worker_pool = WorkerPool(self.num_workers)
        workflows_map: dict[str, WorkflowDAG] = {wf.workflow_id: wf for wf in workflows}

        # Track unmet predecessor counts for each task
        unmet_preds: dict[str, int] = {}
        task_map: dict[str, Task] = {}
        for wf in workflows:
            for task_id, task in wf.tasks.items():
                task_map[task_id] = task
                unmet_preds[task_id] = len(wf.predecessors[task_id])

        # Priority queue for discrete events
        event_queue: list[SimEvent] = []
        event_counter = 0

        # Schedule workflow arrivals
        for wf in workflows:
            heapq.heappush(
                event_queue,
                SimEvent(
                    time=wf.arrival_time,
                    priority=event_counter,
                    event_type=EventType.WORKFLOW_ARRIVAL,
                    payload=wf.workflow_id,
                ),
            )
            event_counter += 1

        ready_queue: list[Task] = []
        event_log: list[dict[str, Any]] = []
        current_time = sim_start_time

        agents = sorted({t.agent_type for t in task_map.values()})
        cache_state: JointCacheState | None = None
        last_access: dict[str, float] = {a: -1.0 for a in agents}
        use_cache = self.cache_capacity is not None and self.cache_mode != "none"
        if use_cache:
            cache_state = JointCacheState(
                capacity_tokens=self.cache_capacity or 1,
                prefix_tokens=self.cache_prefix_tokens,
                agents=agents,
            )

        def dispatch_ready_tasks() -> None:
            nonlocal event_counter
            if not ready_queue or not worker_pool.idle_workers:
                return

            if self.cache_mode == "couloir" or self.joint_couloir:
                decision = decide(
                    list(ready_queue),
                    current_time,
                    workflows_map,
                    cache=cache_state.view() if cache_state else None,
                    weights=self.weights,
                    enable_aging=self.enable_aging,
                )
                ordered_tasks = decision.ordered_tasks
            else:
                ordered_tasks = self.policy_func(
                    list(ready_queue), current_time, workflows_map
                )
            ready_queue.clear()
            ready_queue.extend(ordered_tasks)

            while ready_queue and worker_pool.idle_workers:
                next_task = ready_queue.pop(0)
                idle_worker = worker_pool.idle_workers[0]

                # Cache plan + apply *before* assign so hit/miss can shorten duration.
                cache_meta: dict[str, Any] = {}
                run_s = next_task.actual_duration
                if cache_state is not None:
                    remaining = [next_task, *ready_queue]
                    view = cache_state.view()
                    if self.cache_mode == "couloir" or self.joint_couloir:
                        step = decide(
                            remaining,
                            current_time,
                            workflows_map,
                            cache=view,
                            weights=self.weights,
                            enable_aging=self.enable_aging,
                        )
                        action = step.cache_action
                        cache_meta["utilities"] = {
                            s.task_id: round(s.utility, 4) for s in step.scores
                        }
                    elif self.cache_mode == "ste":
                        action = plan_ste_cache_actions(
                            next_task, remaining, workflows_map, view
                        )
                    elif self.cache_mode == "lru":
                        action = plan_lru_cache_actions(
                            next_task, remaining, view, last_access, current_time
                        )
                    else:
                        action = CacheAction(keep_agents=[next_task.agent_type])
                    cache_meta.update(
                        cache_state.apply_action(action, next_task.agent_type)
                    )
                    hit = bool(cache_meta.get("cache_hit"))
                    run_s = cache_state.effective_duration(next_task.actual_duration, hit)
                    cache_meta["selected_task_id"] = next_task.task_id
                    cache_meta["cache_mode"] = self.cache_mode
                    cache_meta["effective_duration"] = round(run_s, 4)
                    last_access[next_task.agent_type] = current_time

                finish_t = worker_pool.assign_task(
                    idle_worker, next_task, current_time, duration=run_s
                )

                event_log.append(
                    {
                        "time": current_time,
                        "event": "TASK_START",
                        "task_id": next_task.task_id,
                        "agent_type": next_task.agent_type,
                        "worker_id": idle_worker.worker_id,
                        "finish_time": finish_t,
                        "waiting_time": next_task.waiting_time,
                        **({"cache": cache_meta} if cache_meta else {}),
                    }
                )

                heapq.heappush(
                    event_queue,
                    SimEvent(
                        time=finish_t,
                        priority=event_counter,
                        event_type=EventType.TASK_COMPLETE,
                        payload=(next_task.task_id, idle_worker.worker_id),
                    ),
                )
                event_counter += 1

        # Main event loop
        while event_queue:
            event = heapq.heappop(event_queue)
            current_time = event.time

            if event.event_type == EventType.WORKFLOW_ARRIVAL:
                wf_id = event.payload
                wf = workflows_map[wf_id]
                for task_id, task in wf.tasks.items():
                    if unmet_preds[task_id] == 0:
                        task.ready_time = current_time
                        ready_queue.append(task)
                        event_log.append(
                            {
                                "time": current_time,
                                "event": "TASK_READY",
                                "task_id": task_id,
                                "agent_type": task.agent_type,
                            }
                        )
                dispatch_ready_tasks()

            elif event.event_type == EventType.TASK_COMPLETE:
                task_id, worker_id = event.payload
                completed_task = task_map[task_id]
                worker_pool.release_worker(worker_id)

                event_log.append(
                    {
                        "time": current_time,
                        "event": "TASK_FINISH",
                        "task_id": task_id,
                        "agent_type": completed_task.agent_type,
                        "worker_id": worker_id,
                        "duration": completed_task.actual_duration,
                    }
                )

                wf = workflows_map[completed_task.workflow_id]
                for succ_id in wf.successors[task_id]:
                    unmet_preds[succ_id] -= 1
                    if unmet_preds[succ_id] == 0:
                        succ_task = task_map[succ_id]
                        succ_task.ready_time = current_time
                        ready_queue.append(succ_task)
                        event_log.append(
                            {
                                "time": current_time,
                                "event": "TASK_READY",
                                "task_id": succ_id,
                                "agent_type": succ_task.agent_type,
                            }
                        )

                dispatch_ready_tasks()

        cache_stats = None
        if cache_state is not None:
            cache_stats = {
                "hits": cache_state.hits,
                "misses": cache_state.misses,
                "recompute_tokens": cache_state.recompute_tokens,
                "evictions": cache_state.evictions,
                "hit_rate": (
                    cache_state.hits / (cache_state.hits + cache_state.misses)
                    if (cache_state.hits + cache_state.misses)
                    else 0.0
                ),
            }

        metrics = MetricsCollector.calculate(
            policy_name=self.policy_name,
            num_workers=self.num_workers,
            workflows=workflows,
            sim_start_time=sim_start_time,
            cache_stats=cache_stats,
        )

        return SimulationResult(
            policy_name=self.policy_name,
            num_workers=self.num_workers,
            workflows=list(workflows),
            metrics=metrics,
            event_log=event_log,
            cache_capacity=self.cache_capacity,
        )
