"""Discrete-event simulation engine for multi-agent DAG scheduling across worker pools."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Sequence

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

    def assign_task(self, worker: Worker, task: Task, current_time: float) -> float:
        """Assign task to worker, return finish timestamp."""
        worker.current_task = task
        task.assigned_worker = worker.worker_id
        task.start_time = current_time
        task.finish_time = current_time + task.actual_duration
        worker.busy_until = task.finish_time
        worker.total_busy_time += task.actual_duration
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


# Policy sort function signature: (ready_queue: list[Task], current_time: float, workflows_map: dict[str, WorkflowDAG]) -> list[Task]
PolicyFunction = Callable[[list[Task], float, dict[str, WorkflowDAG]], list[Task]]


class SimulationEngine:
    """Discrete-event simulator for multi-agent DAG task execution under worker contention."""

    def __init__(
        self,
        num_workers: int,
        policy_func: PolicyFunction,
        policy_name: str = "custom",
    ) -> None:
        self.num_workers = num_workers
        self.policy_func = policy_func
        self.policy_name = policy_name

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

        def dispatch_ready_tasks() -> None:
            nonlocal event_counter
            if not ready_queue or not worker_pool.idle_workers:
                return

            # Apply policy ordering to ready queue
            ordered_tasks = self.policy_func(list(ready_queue), current_time, workflows_map)
            ready_queue.clear()
            ready_queue.extend(ordered_tasks)

            while ready_queue and worker_pool.idle_workers:
                next_task = ready_queue.pop(0)
                idle_worker = worker_pool.idle_workers[0]
                finish_t = worker_pool.assign_task(idle_worker, next_task, current_time)

                event_log.append(
                    {
                        "time": current_time,
                        "event": "TASK_START",
                        "task_id": next_task.task_id,
                        "agent_type": next_task.agent_type,
                        "worker_id": idle_worker.worker_id,
                        "finish_time": finish_t,
                        "waiting_time": next_task.waiting_time,
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

        metrics = MetricsCollector.calculate(
            policy_name=self.policy_name,
            num_workers=self.num_workers,
            workflows=workflows,
            sim_start_time=sim_start_time,
        )

        return SimulationResult(
            policy_name=self.policy_name,
            num_workers=self.num_workers,
            workflows=list(workflows),
            metrics=metrics,
            event_log=event_log,
        )
