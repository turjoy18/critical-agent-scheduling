"""Agent Step Graph and steps-to-execution (KVFlow §3.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import networkx as nx

AggKind = Literal["max", "min"]

# Far-future / unreachable agents: preferred eviction candidates.
UNREACHABLE = 10**9


@dataclass
class AgentStepGraph:
    """Dependency graph of agent invocations with per-node step aggregators."""

    nodes: list[str]
    edges: list[tuple[str, str]]
    aggregators: dict[str, AggKind] = field(default_factory=dict)

    def __post_init__(self) -> None:
        known = set(self.nodes)
        for u, v in self.edges:
            if u not in known or v not in known:
                raise ValueError(f"Edge ({u}, {v}) references unknown node")
        for node in self.nodes:
            self.aggregators.setdefault(node, "max")

    def to_networkx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from(self.edges)
        return g

    def set_aggregator(self, node: str, kind: AggKind) -> None:
        if node not in self.nodes:
            raise KeyError(node)
        self.aggregators[node] = kind


def steps_to_execution(
    graph: AgentStepGraph,
    active_agents: Iterable[str],
) -> dict[str, int]:
    """Compute how soon each agent is expected to run.

    Active agents receive STE 0. Downstream STE is
    ``agg(predecessor STEs) + 1``, where ``agg`` is ``max`` (AND-join)
    or ``min`` (OR-branch). Agents not reachable as future work from any
    active agent keep the ``UNREACHABLE`` sentinel.
    """
    active = set(active_agents)
    unknown = active - set(graph.nodes)
    if unknown:
        raise ValueError(f"Active agents not in graph: {sorted(unknown)}")

    ste: dict[str, int] = {n: UNREACHABLE for n in graph.nodes}
    for agent in active:
        ste[agent] = 0

    g = graph.to_networkx()
    for node in nx.topological_sort(g):
        if node in active:
            continue
        preds = list(g.predecessors(node))
        if not preds:
            continue
        finite = [ste[p] for p in preds if ste[p] < UNREACHABLE]
        if not finite:
            continue
        agg = max if graph.aggregators[node] == "max" else min
        ste[node] = agg(finite) + 1

    return ste
