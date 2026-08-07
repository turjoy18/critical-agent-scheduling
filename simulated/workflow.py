import json
import random
import time
import uuid
from collections.abc import Callable
from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

# --------------------------- State ---------------------------

def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}


class TraceEvent(TypedDict):
    workflow_id: str
    agent: str
    parents: list[str]
    start_time: float
    end_time: float
    runtime_s: float
    input_tokens: int
    output_tokens: int


class WorkflowState(TypedDict):
    workflow_id: str
    task: str
    plan: str
    research: Annotated[dict[str, str], merge_dicts]
    synthesis: str
    critique: str
    final_answer: str
    trace: Annotated[list[TraceEvent], add]


# --------------------------- Tracing Wrapper ---------------------------

def traced_node(
    agent_name: str,
    parents: list[str],
    fn: Callable[[WorkflowState], tuple[dict, dict]],
):
    def node(state: WorkflowState) -> dict:
        start = time.perf_counter()
        output_update, usage = fn(state)
        end = time.perf_counter()

        event: TraceEvent = {
            "workflow_id": state["workflow_id"],
            "agent": agent_name,
            "parents": parents,
            "start_time": start,
            "end_time": end,
            "runtime_s": end - start,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
        }

        return {
            **output_update,
            "trace": [event],
        }

    return node


# --------------------------- Simulated agents ---------------------------

def planner_agent(state: WorkflowState):
    time.sleep(random.uniform(0.5, 1.0))
    return {"plan": f"Plan for {state['task']}"}, {}

def researcher_a_agent(state: WorkflowState):
    time.sleep(random.uniform(1.0, 2.0))
    return {"research": {"researcher_a": "Findings from researcher A"}}, {}

def researcher_b_agent(state: WorkflowState):
    # Intentionally slower so it often becomes critical
    time.sleep(random.uniform(2.5, 4.0))
    return {"research": {"researcher_b": "Findings from researcher B"}}, {}

def researcher_c_agent(state: WorkflowState):
    time.sleep(random.uniform(1.0, 2.5))
    return {"research": {"researcher_c": "Findings from researcher C"}}, {}

def synthesizer_agent(state: WorkflowState):
    time.sleep(random.uniform(0.8, 1.0))
    combined = "\n".join(state["research"].values())
    return {"synthesis": f"Synthesis:\n{combined}"}, {}

def critic_agent(state: WorkflowState):
    time.sleep(random.uniform(0.7, 1.2))
    return {"critique": f"Critique of the synthesis: {state['synthesis'][:80]}"}, {}

def final_writer_agent(state: WorkflowState):
    time.sleep(random.uniform(0.7, 1.2))
    return {"final_answer": f"Final answer based on critique: {state['critique']}"}, {}


# --------------------------- Build Graph ---------------------------

builder = StateGraph(WorkflowState)
builder.add_node("planner", traced_node("planner", [], planner_agent))
builder.add_node("researcher_a", traced_node("researcher_a", ["planner"], researcher_a_agent))
builder.add_node("researcher_b", traced_node("researcher_b", ["planner"], researcher_b_agent))
builder.add_node("researcher_c", traced_node("researcher_c", ["planner"], researcher_c_agent))
builder.add_node(
    "synthesizer",
    traced_node("synthesizer", ["researcher_a", "researcher_b", "researcher_c"], synthesizer_agent),
)
builder.add_node("critic", traced_node("critic", ["synthesizer"], critic_agent))
builder.add_node("final_writer", traced_node("final_writer", ["critic"], final_writer_agent))

builder.add_edge(START, "planner")
builder.add_edge("planner", "researcher_a")
builder.add_edge("planner", "researcher_b")
builder.add_edge("planner", "researcher_c")
builder.add_edge(["researcher_a", "researcher_b", "researcher_c"], "synthesizer")
builder.add_edge("synthesizer", "critic")
builder.add_edge("critic", "final_writer")
builder.add_edge("final_writer", END)

graph = builder.compile()

# --------------------------- Run experiments ---------------------------

TASKS = [
    "Explain KV caching for multi-agent LLM workflows.",
    "Compare LangGraph and AutoGen.",
    "Summarize DAG scheduling methods.",
    "Describe critical path analysis.",
    "Explain HEFT scheduling.",
]


def main(num_runs: int = 20):
    all_traces = []

    for i in range(num_runs):
        task = TASKS[i % len(TASKS)]
        workflow_id = str(uuid.uuid4())

        print(f"Running workflow {i + 1}/{num_runs}: {task}")

        result = graph.invoke(
            {
                "workflow_id": workflow_id,
                "task": task,
                "plan": "",
                "research": {},
                "synthesis": "",
                "critique": "",
                "final_answer": "",
                "trace": [],
            }
        )

        all_traces.append(
            {
                "workflow_id": workflow_id,
                "task": task,
                "trace": result["trace"],
                "final_answer": result["final_answer"],
            }
        )

    with open("traces.jsonl", "w", encoding="utf-8") as f:
        for row in all_traces:
            f.write(json.dumps(row) + "\n")

    print(f"Saved {len(all_traces)} run to traces.jsonl")


if __name__ == "__main__":
    main(num_runs=20)