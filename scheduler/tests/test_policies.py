"""Unit tests for scheduling policies."""

from __future__ import annotations

import unittest

from scheduler.dag import create_research_writing_workflow
from scheduler.policies import (
    critical_path_first_policy,
    fifo_policy,
    get_policy,
    heft_policy,
    least_slack_first_policy,
)


class TestSchedulingPolicies(unittest.TestCase):
    def setUp(self) -> None:
        self.wf = create_research_writing_workflow(workflow_id="wf0", profile="llm")
        self.workflows = {"wf0": self.wf}
        # Simulate researchers becoming ready at t = 21.40
        self.r_a = self.wf.tasks["wf0_researcher_a"]
        self.r_b = self.wf.tasks["wf0_researcher_b"]
        self.r_c = self.wf.tasks["wf0_researcher_c"]

        self.r_a.ready_time = 21.40
        self.r_b.ready_time = 21.40
        self.r_c.ready_time = 21.40

    def test_cpf_prioritizes_researcher_b(self) -> None:
        ready = [self.r_a, self.r_b, self.r_c]
        sorted_tasks = critical_path_first_policy(ready, current_time=21.40, workflows=self.workflows)
        # researcher_b has highest upward rank (97.35) vs c (75.60) vs a (66.73)
        self.assertEqual(sorted_tasks[0].agent_type, "researcher_b")
        self.assertEqual(sorted_tasks[1].agent_type, "researcher_c")
        self.assertEqual(sorted_tasks[2].agent_type, "researcher_a")

    def test_slack_prioritizes_lowest_slack(self) -> None:
        ready = [self.r_a, self.r_c, self.r_b]
        sorted_tasks = least_slack_first_policy(ready, current_time=21.40, workflows=self.workflows)
        # researcher_b has 0 slack -> top priority
        self.assertEqual(sorted_tasks[0].agent_type, "researcher_b")
        # researcher_c has 21.75s slack -> second
        self.assertEqual(sorted_tasks[1].agent_type, "researcher_c")
        # researcher_a has 30.62s slack -> third
        self.assertEqual(sorted_tasks[2].agent_type, "researcher_a")

    def test_heft_prioritizes_highest_upward_rank(self) -> None:
        ready = [self.r_a, self.r_b, self.r_c]
        sorted_tasks = heft_policy(ready, current_time=21.40, workflows=self.workflows)
        self.assertEqual(sorted_tasks[0].agent_type, "researcher_b")
        self.assertEqual(sorted_tasks[1].agent_type, "researcher_c")
        self.assertEqual(sorted_tasks[2].agent_type, "researcher_a")

    def test_fifo_preserves_arrival_order(self) -> None:
        self.r_a.ready_time = 10.0
        self.r_b.ready_time = 20.0
        self.r_c.ready_time = 15.0
        ready = [self.r_c, self.r_b, self.r_a]
        sorted_tasks = fifo_policy(ready, current_time=25.0, workflows=self.workflows)
        self.assertEqual(sorted_tasks[0].agent_type, "researcher_a")
        self.assertEqual(sorted_tasks[1].agent_type, "researcher_c")
        self.assertEqual(sorted_tasks[2].agent_type, "researcher_b")

    def test_get_policy_lookup(self) -> None:
        self.assertIsNotNone(get_policy("fifo"))
        self.assertIsNotNone(get_policy("cpf"))
        self.assertIsNotNone(get_policy("slack"))
        self.assertIsNotNone(get_policy("heft"))
        self.assertIsNotNone(get_policy("couloir"))
        with self.assertRaises(ValueError):
            get_policy("nonexistent")


if __name__ == "__main__":
    unittest.main()
