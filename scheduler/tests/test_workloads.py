"""Tests for multi-DAG workload builders."""

from __future__ import annotations

import unittest

from scheduler.workloads import create_workflow, dag_agent_means


class TestWorkloads(unittest.TestCase):
    def test_writing_dag(self) -> None:
        wf = create_workflow("writing", workflow_id="w")
        self.assertEqual(len(wf.tasks), 7)
        self.assertIn("w_researcher_b", wf.tasks)

    def test_chain_is_linear(self) -> None:
        wf = create_workflow("chain", workflow_id="c")
        self.assertEqual(len(wf.tasks), 7)
        for tid, preds in wf.predecessors.items():
            self.assertLessEqual(len(preds), 1)
        for tid, succs in wf.successors.items():
            self.assertLessEqual(len(succs), 1)
        path, length = wf.compute_critical_path(use_expected=True)
        self.assertEqual(len(path), 7)
        self.assertAlmostEqual(length, sum(dag_agent_means("chain").values()), places=5)

    def test_fanout_parallelism(self) -> None:
        wf = create_workflow("fanout", workflow_id="f")
        planner = "f_planner"
        self.assertEqual(len(wf.successors[planner]), 6)
        joiner = "f_joiner"
        self.assertEqual(len(wf.predecessors[joiner]), 6)

    def test_unknown_dag(self) -> None:
        with self.assertRaises(ValueError):
            create_workflow("nope")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
