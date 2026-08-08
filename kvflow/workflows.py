"""Workflow topologies used by the KVFlow conceptual reproduction."""

from __future__ import annotations

from kvflow.step_graph import AgentStepGraph

# Same DAG as simulated/analyze.py and llm/analyze_llm.py.
RESEARCH_WRITING_EDGES: list[tuple[str, str]] = [
    ("planner", "researcher_a"),
    ("planner", "researcher_b"),
    ("planner", "researcher_c"),
    ("researcher_a", "synthesizer"),
    ("researcher_b", "synthesizer"),
    ("researcher_c", "synthesizer"),
    ("synthesizer", "critic"),
    ("critic", "final_writer"),
]

RESEARCH_WRITING_NODES: list[str] = [
    "planner",
    "researcher_a",
    "researcher_b",
    "researcher_c",
    "synthesizer",
    "critic",
    "final_writer",
]


def research_writing_graph() -> AgentStepGraph:
    """Profiling DAG: planner → parallel researchers → synthesizer → critic → writer.

    Synthesizer is an AND-join (aggregator ``max``).
    """
    return AgentStepGraph(
        nodes=list(RESEARCH_WRITING_NODES),
        edges=list(RESEARCH_WRITING_EDGES),
        aggregators={"synthesizer": "max"},
    )


def or_branch_toy_graph() -> AgentStepGraph:
    """Asymmetric OR/AND toy used to contrast ``min`` vs ``max`` joins.

    Topology::

        A → B → C → Join
        A → D ──────┘

    With A active: B=1, C=2, D=1.
    Join with max → 3; with min → 2.
    """
    nodes = ["A", "B", "C", "D", "Join"]
    edges = [
        ("A", "B"),
        ("B", "C"),
        ("C", "Join"),
        ("A", "D"),
        ("D", "Join"),
    ]
    return AgentStepGraph(nodes=nodes, edges=edges, aggregators={"Join": "min"})


def sequential_agents_graph(n: int = 10) -> AgentStepGraph:
    """Linear chain of ``n`` agents (KVFlow single-workflow microbench shape)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    nodes = [f"agent_{i}" for i in range(n)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(n - 1)]
    return AgentStepGraph(nodes=nodes, edges=edges)
