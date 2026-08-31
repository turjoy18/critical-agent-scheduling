"""Unit tests for discrete-event SimulationEngine."""

from __future__ import annotations

import unittest

from scheduler.dag import create_research_writing_workflow
from scheduler.engine import SimulationEngine
from scheduler.policies import (
    critical_path_first_policy,
    fifo_policy,
    heft_policy,
    least_slack_first_policy,
)


class TestSimulationEngine(unittest.TestCase):
    def test_single_workflow_infinite_concurrency(self) -> None:
        # K=3 is full concurrency for 3 parallel researchers
        wf = create_research_writing_workflow(workflow_id="wf0", profile="llm")
        engine = SimulationEngine(num_workers=3, policy_func=critical_path_first_policy, policy_name="cpf")
        result = engine.run([wf])

        # Theoretical optimal critical path: 118.75s
        self.assertAlmostEqual(result.metrics.makespan, 118.75, places=2)
        self.assertAlmostEqual(result.metrics.critical_path_delay, 0.0, places=2)

    def test_single_workflow_contention_k2_cpf_vs_worst_fifo(self) -> None:
        # Under K=2, CPF picks B (41.47s) and C (19.72s), then A (10.85s) runs on the freed slot.
        # Makespan matches ideal critical path = 118.75s.
        wf_cpf = create_research_writing_workflow(workflow_id="wf0", profile="llm")
        engine_cpf = SimulationEngine(num_workers=2, policy_func=critical_path_first_policy, policy_name="cpf")
        res_cpf = engine_cpf.run([wf_cpf])

        self.assertAlmostEqual(res_cpf.metrics.makespan, 118.75, places=2)

        # Inverted FIFO where A and C are dispatched before B:
        # B is delayed by 10.85s (A's duration), making makespan 118.75 + 10.85 = 129.60s.
        def reverse_critical_policy(ready, curr_t, wfs):
            return sorted(ready, key=lambda t: t.expected_duration)

        wf_rev = create_research_writing_workflow(workflow_id="wf0", profile="llm")
        engine_rev = SimulationEngine(num_workers=2, policy_func=reverse_critical_policy, policy_name="rev")
        res_rev = engine_rev.run([wf_rev])

        self.assertAlmostEqual(res_rev.metrics.makespan, 129.60, places=2)
        self.assertTrue(res_cpf.metrics.makespan < res_rev.metrics.makespan)

    def test_batch_workflows_heft_and_slack(self) -> None:
        # 3 concurrent workflows under K=2 workers
        wfs_heft = [
            create_research_writing_workflow(workflow_id=f"wf{i}", profile="simulated", arrival_time=0.0)
            for i in range(3)
        ]
        engine_heft = SimulationEngine(num_workers=2, policy_func=heft_policy, policy_name="heft")
        res_heft = engine_heft.run(wfs_heft)

        self.assertGreater(res_heft.metrics.makespan, 0.0)
        self.assertEqual(res_heft.metrics.num_tasks, 21)
        self.assertEqual(res_heft.metrics.num_workflows, 3)
        self.assertGreater(res_heft.metrics.worker_utilization, 0.8)


if __name__ == "__main__":
    unittest.main()
