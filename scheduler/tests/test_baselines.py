"""Tests for two-phase CPF→STE and baseline matrix."""

from __future__ import annotations

import unittest

from scheduler.baselines import plan_ste_cache_actions
from scheduler.couloir import CacheView
from scheduler.engine import SimulationEngine
from scheduler.policies import get_policy
from scheduler.workloads import create_workflow


class TestBaselines(unittest.TestCase):
    def test_two_phase_matches_cpf_order(self) -> None:
        wf = create_workflow("writing", workflow_id="wf0")
        wfs = {"wf0": wf}
        ready = [
            wf.tasks["wf0_researcher_a"],
            wf.tasks["wf0_researcher_b"],
            wf.tasks["wf0_researcher_c"],
        ]
        for t in ready:
            t.ready_time = 21.4
        a = get_policy("two_phase")(ready, 21.4, wfs)
        b = get_policy("cpf")(ready, 21.4, wfs)
        self.assertEqual([t.agent_type for t in a], [t.agent_type for t in b])

    def test_ste_evicts_far_agents(self) -> None:
        wf = create_workflow("writing", workflow_id="wf0")
        wfs = {"wf0": wf}
        ready = [wf.tasks["wf0_researcher_b"]]
        ready[0].ready_time = 21.4
        cache = CacheView(
            capacity_tokens=40,
            gpu_tokens=40,
            agent_on_gpu={"planner": True, "final_writer": True},
            agent_prefix_tokens={"planner": 20, "final_writer": 20, "researcher_b": 20},
        )
        action = plan_ste_cache_actions(ready[0], ready, wfs, cache)
        self.assertIn("researcher_b", action.keep_agents)
        # Upstream planner is unreachable from active researcher_b → highest STE → evict first
        self.assertTrue(action.evict_agents)
        self.assertEqual(action.evict_agents[0], "planner")

    def test_two_phase_engine_with_cache(self) -> None:
        wf = create_workflow("writing", workflow_id="wf0")
        engine = SimulationEngine(
            num_workers=2,
            policy_func=get_policy("two_phase"),
            policy_name="two_phase",
            cache_capacity=40,
            cache_prefix_tokens=20,
            cache_mode="ste",
        )
        res = engine.run([wf])
        self.assertGreater(res.metrics.makespan, 0.0)
        self.assertIsNotNone(res.metrics.cache_hit_rate)
        modes = {
            e["cache"]["cache_mode"]
            for e in res.event_log
            if e.get("cache") and "cache_mode" in e["cache"]
        }
        self.assertIn("ste", modes)


if __name__ == "__main__":
    unittest.main()
