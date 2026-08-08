"""Unit tests for Agent Step Graph and steps-to-execution."""

from __future__ import annotations

import unittest

from kvflow.step_graph import UNREACHABLE, AgentStepGraph, steps_to_execution
from kvflow.workflows import or_branch_toy_graph, research_writing_graph


class TestResearchWritingSTE(unittest.TestCase):
    def test_planner_active_expected_stes(self) -> None:
        graph = research_writing_graph()
        ste = steps_to_execution(graph, active_agents=["planner"])
        self.assertEqual(ste["planner"], 0)
        self.assertEqual(ste["researcher_a"], 1)
        self.assertEqual(ste["researcher_b"], 1)
        self.assertEqual(ste["researcher_c"], 1)
        self.assertEqual(ste["synthesizer"], 2)
        self.assertEqual(ste["critic"], 3)
        self.assertEqual(ste["final_writer"], 4)

    def test_synthesizer_is_and_join_max(self) -> None:
        graph = research_writing_graph()
        self.assertEqual(graph.aggregators["synthesizer"], "max")
        # Mid-workflow: only researchers active → synthesizer waits for all three.
        ste = steps_to_execution(
            graph,
            active_agents=["researcher_a", "researcher_b", "researcher_c"],
        )
        self.assertEqual(ste["researcher_a"], 0)
        self.assertEqual(ste["synthesizer"], 1)
        self.assertEqual(ste["critic"], 2)
        self.assertEqual(ste["planner"], UNREACHABLE)


class TestOrBranchToy(unittest.TestCase):
    def test_min_join_smaller_than_max(self) -> None:
        graph = or_branch_toy_graph()
        ste_min = steps_to_execution(graph, active_agents=["A"])
        self.assertEqual(ste_min["B"], 1)
        self.assertEqual(ste_min["C"], 2)
        self.assertEqual(ste_min["D"], 1)
        self.assertEqual(ste_min["Join"], 2)  # min(2, 1) + 1

        graph_max = AgentStepGraph(
            nodes=list(graph.nodes),
            edges=list(graph.edges),
            aggregators={"Join": "max"},
        )
        ste_max = steps_to_execution(graph_max, active_agents=["A"])
        self.assertEqual(ste_max["Join"], 3)  # max(2, 1) + 1
        self.assertLess(ste_min["Join"], ste_max["Join"])


class TestUnreachableAndMultiActive(unittest.TestCase):
    def test_unreachable_ancestors(self) -> None:
        graph = research_writing_graph()
        ste = steps_to_execution(graph, active_agents=["critic"])
        self.assertEqual(ste["critic"], 0)
        self.assertEqual(ste["final_writer"], 1)
        self.assertEqual(ste["planner"], UNREACHABLE)
        self.assertEqual(ste["synthesizer"], UNREACHABLE)

    def test_unknown_active_raises(self) -> None:
        graph = research_writing_graph()
        with self.assertRaises(ValueError):
            steps_to_execution(graph, active_agents=["no_such_agent"])


if __name__ == "__main__":
    unittest.main()
