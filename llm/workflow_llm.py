import json
import os
import time
import uuid
from collections.abc import Callable
from operator import add
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv(Path(__file__).parent.parent / ".env")


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


# --------------------------- OpenRouter client ---------------------------

llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    temperature=0.2,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "critical-agent-scheduling",
    },
)


def call_llm(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
    response = llm.invoke(
        [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
    )

    usage_meta = getattr(response, "usage_metadata", None) or {}
    usage = {
        "input_tokens": usage_meta.get("input_tokens", 0),
        "output_tokens": usage_meta.get("output_tokens", 0),
    }

    content = response.content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    return content, usage


# --------------------------- Tracing wrapper ---------------------------

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
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }

        return {**output_update, "trace": [event]}

    return node


# --------------------------- LLM agents ---------------------------

def planner_agent(state: WorkflowState):
    content, usage = call_llm(
        "You are a planning agent. Produce a short research plan with 3 bullet points.",
        state["task"],
    )
    return {"plan": content}, usage

def researcher_a_agent(state: WorkflowState):
    content, usage = call_llm(
        "You are Researcher A. Focus on definitions and background. Keep answer under 120 words.",
        f"Task: {state['task']}\nPlan:\n{state['plan']}",
    )
    return {"research": {"researcher_a": content}}, usage

def researcher_b_agent(state: WorkflowState):
    content, usage = call_llm(
        "You are Researcher B. Provide deeper technical analysis with examples and tradeoffs. Keep answer under 220 words.",
        f"Task: {state['task']}\nPlan:\n{state['plan']}",
    )
    return {"research": {"researcher_b": content}}, usage

def researcher_c_agent(state: WorkflowState):
    content, usage = call_llm(
        "You are Researcher C. Focus on practical implications and limitations. Keep answer under 120 words.",
        f"Task: {state['task']}\nPlan:\n{state['plan']}",
    )
    return {"research": {"researcher_c": content}}, usage

def synthesizer_agent(state: WorkflowState):
    combined = "\n\n".join(
        f"{name}:\n{text}" for name, text in state["research"].items()
    )
    content, usage = call_llm(
        "You are a synthesizer. Merge the research notes into one coherent summary under 180 words.",
        combined,
    )
    return {"synthesis": content}, usage

def critic_agent(state: WorkflowState):
    content, usage = call_llm(
        "You are a critic. Point out 2 weaknesses and 1 improvement. Keep under 100 words.",
        state["synthesis"],
    )
    return {"critique": content}, usage

def final_writer_agent(state: WorkflowState):
    content, usage = call_llm(
        "You are the final writer. Produce a polished final answer under 180 words using the synthesis and critique.",
        f"Synthesis:\n{state['synthesis']}\n\nCritique:\n{state['critique']}",
    )
    return {"final_answer": content}, usage


# --------------------------- Build graph ---------------------------

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

def main(num_runs: int = 5):
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY not found.")

    # Having suffered the tragedy of a crash mid-run due to exceeding the rate limit per minute for free tier, I now take precautions.
    # It is not strictly necessary, but it's a good idea to avoid clobbering existing traces.
    # This could be done more elegantly with a check for the file's size, but this is a simple solution for now.

    traces_path = "traces_llm.jsonl"
    # Clear previous traces so a crash mid-run still keeps completed ones.
    open(traces_path, "w", encoding="utf-8").close()

    saved = 0
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

        row = {
            "workflow_id": workflow_id,
            "task": task,
            "trace": result["trace"],
            "final_answer": result["final_answer"],
        }

        # Append after each success so rate-limit crashes don't lose finished runs.
        with open(traces_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        saved += 1
        print(f"Saved run {saved}/{num_runs} to {traces_path}")

        # Free tier for OpenRouter is 20 req/min; each workflow uses ~7 calls, so pause between runs.
        if i < num_runs - 1:
            pause_s = 20
            print(f"Waiting {pause_s}s to stay under free-models-per-min limit...")
            time.sleep(pause_s)

    print(f"Done. Saved {saved} traces to {traces_path}")


if __name__ == "__main__":
    main(num_runs=5)