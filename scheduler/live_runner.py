"""Live LLM workflow runner with policy-based dispatch (free-tier backends)."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from scheduler.dag import create_research_writing_workflow
from scheduler.llm_backend import LLMBackend, get_backend
from scheduler.policies import get_policy

load_dotenv()

DEFAULT_OUT_DIR = Path("scheduler/results")

TASKS = [
    "Explain KV caching for multi-agent LLM workflows.",
    "Compare LangGraph and AutoGen.",
    "Summarize DAG scheduling methods in distributed systems.",
    "Describe critical path analysis in parallel workflows.",
]


class LiveWorkflowExecutor:
    """Executes live multi-agent DAGs under worker slot contention and policy dispatch."""

    def __init__(
        self,
        num_workers: int = 2,
        policy_name: str = "cpf",
        backend: LLMBackend | None = None,
        backend_name: str | None = None,
    ) -> None:
        self.num_workers = num_workers
        self.policy_name = policy_name
        self.policy_func = get_policy(policy_name)
        self.backend = backend or get_backend(backend_name)
        self.model = self.backend.model

    def run_workflow(self, task_prompt: str, workflow_id: str | None = None) -> dict[str, Any]:
        wf_id = workflow_id or f"live_{uuid.uuid4().hex[:8]}"
        wf_dag = create_research_writing_workflow(workflow_id=wf_id, profile="llm")
        workflows_map = {wf_id: wf_dag}

        state: dict[str, Any] = {
            "task": task_prompt,
            "plan": "",
            "research": {},
            "synthesis": "",
            "critique": "",
            "final_answer": "",
        }

        traces: list[dict[str, Any]] = []
        traces_lock = Lock()
        t0 = time.perf_counter()

        def execute_agent(agent_type: str) -> dict[str, Any]:
            task_obj = wf_dag.tasks[f"{wf_id}_{agent_type}"]
            t_start = time.perf_counter()
            task_obj.start_time = t_start - t0

            prompts = {
                "planner": (
                    "You are a planning agent. Produce a concise research plan with 3 bullet points.",
                    state["task"],
                ),
                "researcher_a": (
                    "You are Researcher A. Focus on definitions and background. Keep answer under 100 words.",
                    f"Task: {state['task']}\nPlan:\n{state['plan']}",
                ),
                "researcher_b": (
                    "You are Researcher B. Provide a comprehensive technical deep-dive with architectural tradeoffs, equations or system details. Keep answer under 250 words.",
                    f"Task: {state['task']}\nPlan:\n{state['plan']}",
                ),
                "researcher_c": (
                    "You are Researcher C. Focus on practical limitations and real-world system constraints. Keep under 100 words.",
                    f"Task: {state['task']}\nPlan:\n{state['plan']}",
                ),
                "synthesizer": (
                    "You are a synthesizer. Merge research notes into one coherent technical summary under 160 words.",
                    "\n\n".join(f"{n}:\n{t}" for n, t in state["research"].items()),
                ),
                "critic": (
                    "You are a critic. Point out 2 technical weaknesses and 1 improvement. Keep under 90 words.",
                    state["synthesis"],
                ),
                "final_writer": (
                    "You are the final writer. Produce a polished final report using the synthesis and critique under 160 words.",
                    f"Synthesis:\n{state['synthesis']}\n\nCritique:\n{state['critique']}",
                ),
            }
            if agent_type not in prompts:
                raise ValueError(f"Unknown agent type {agent_type}")
            system, user = prompts[agent_type]
            resp = self.backend.complete(system, user)
            content, usage = resp.text, resp.usage

            if agent_type == "planner":
                state["plan"] = content
            elif agent_type.startswith("researcher_"):
                state["research"][agent_type] = content
            elif agent_type == "synthesizer":
                state["synthesis"] = content
            elif agent_type == "critic":
                state["critique"] = content
            elif agent_type == "final_writer":
                state["final_answer"] = content

            t_end = time.perf_counter()
            task_obj.finish_time = t_end - t0
            task_obj.actual_duration = t_end - t_start

            record = {
                "workflow_id": wf_id,
                "agent": agent_type,
                "start_time": task_obj.start_time,
                "finish_time": task_obj.finish_time,
                "runtime_s": task_obj.actual_duration,
                "waiting_time": task_obj.waiting_time,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "provider": resp.provider,
                "model": resp.model,
            }
            with traces_lock:
                traces.append(record)
            return record

        # Thread pool with K worker slots
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            # 1. Planner
            wf_dag.tasks[f"{wf_id}_planner"].ready_time = 0.0
            fut = pool.submit(execute_agent, "planner")
            fut.result()

            # 2. Parallel Researchers with Contention & Policy Ordering
            researchers = ["researcher_a", "researcher_b", "researcher_c"]
            r_tasks = [wf_dag.tasks[f"{wf_id}_{r}"] for r in researchers]
            t_r_ready = time.perf_counter() - t0
            for t in r_tasks:
                t.ready_time = t_r_ready

            # Sort ready researchers using policy
            ordered_tasks = self.policy_func(r_tasks, t_r_ready, workflows_map)
            futures: list[Future] = [
                pool.submit(execute_agent, t.agent_type) for t in ordered_tasks
            ]
            for f in futures:
                f.result()

            # 3. Synthesizer
            t_s_ready = time.perf_counter() - t0
            wf_dag.tasks[f"{wf_id}_synthesizer"].ready_time = t_s_ready
            pool.submit(execute_agent, "synthesizer").result()

            # 4. Critic
            t_c_ready = time.perf_counter() - t0
            wf_dag.tasks[f"{wf_id}_critic"].ready_time = t_c_ready
            pool.submit(execute_agent, "critic").result()

            # 5. Final Writer
            t_w_ready = time.perf_counter() - t0
            wf_dag.tasks[f"{wf_id}_final_writer"].ready_time = t_w_ready
            pool.submit(execute_agent, "final_writer").result()

        total_makespan = time.perf_counter() - t0

        return {
            "workflow_id": wf_id,
            "policy": self.policy_name,
            "num_workers": self.num_workers,
            "makespan": round(total_makespan, 4),
            "traces": traces,
            "final_answer": state["final_answer"],
        }


def run_live_comparison(
    num_runs: int = 5,
    num_workers: int = 2,
    out_dir: Path = DEFAULT_OUT_DIR,
    pause_s: float = 4.0,
    backend_name: str | None = None,
) -> pd.DataFrame:
    """Run live comparative evaluation of policies on a free-tier backend."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    traces_file = out_dir / "live_traces.jsonl"
    open(traces_file, "w", encoding="utf-8").close()

    backend = get_backend(backend_name)
    print(f"Using free-tier backend: {backend.provider} / {backend.model}")
    policies = ["cpf", "fifo", "couloir"]

    for i in range(num_runs):
        task = TASKS[i % len(TASKS)]
        print(f"\n--- Live Task {i + 1}/{num_runs}: {task} ---")

        for pol in policies:
            print(f"Running Policy [{pol.upper()}] with {num_workers} worker slots...")
            executor = LiveWorkflowExecutor(
                num_workers=num_workers,
                policy_name=pol,
                backend=backend,
            )
            wf_result = executor.run_workflow(task)
            print(f"-> Completed [{pol.upper()}] in {wf_result['makespan']}s")

            with open(traces_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(wf_result) + "\n")

            results.append(
                {
                    "task": task,
                    "policy": pol,
                    "num_workers": num_workers,
                    "makespan_s": wf_result["makespan"],
                    "provider": backend.provider,
                    "model": backend.model,
                }
            )
            time.sleep(pause_s)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "live_comparison.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run live free-tier LLM workflow scheduling validation."
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of workflow runs")
    parser.add_argument("--workers", type=int, default=2, help="Worker pool slots")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["groq", "gemini", "openrouter"],
        help="Free-tier backend (default: first available key)",
    )
    args = parser.parse_args()

    df = run_live_comparison(
        num_runs=args.runs,
        num_workers=args.workers,
        out_dir=Path(args.out),
        backend_name=args.backend,
    )
    print("\n=== Live Results Summary ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
