"""Integration tests for Couloir joint compute + cache in SimulationEngine."""

from __future__ import annotations

import unittest

from scheduler.engine import SimulationEngine
from scheduler.policies import get_policy
from scheduler.workloads import create_workflow


class TestJointCouloir(unittest.TestCase):
    def test_couloir_policy_registered(self) -> None:
        self.assertIsNotNone(get_policy("couloir"))

    def test_couloir_under_k2_contention(self) -> None:
        wf = create_workflow("writing", workflow_id="wf0")
        engine = SimulationEngine(
            num_workers=2,
            policy_func=get_policy("couloir"),
            policy_name="couloir",
            cache_capacity=2048,
            cache_prefix_tokens=512,
        )
        res = engine.run([wf])
        self.assertGreater(res.metrics.makespan, 0.0)
        # Prefetch hits shorten prefill → makespan ≤ cold critical path.
        self.assertLessEqual(res.metrics.makespan, 118.75 + 1e-6)
        starts = [e for e in res.event_log if e["event"] == "TASK_START"]
        self.assertTrue(any("cache" in e for e in starts))
        self.assertIsNotNone(res.metrics.cache_hit_rate)
        self.assertIsNotNone(res.metrics.recompute_tokens)
        self.assertGreater(res.metrics.cache_hit_rate or 0.0, 0.0)

    def test_cache_hit_reduces_makespan_vs_lru(self) -> None:
        def _run(mode: str, pol: str) -> float:
            wf = create_workflow("fanout", workflow_id="f0")
            eng = SimulationEngine(
                num_workers=2,
                policy_func=get_policy(pol),
                policy_name=pol,
                cache_capacity=1024,
                cache_prefix_tokens=512,
                cache_mode=mode,
                joint_couloir=(mode == "couloir"),
            )
            return eng.run([wf]).metrics.makespan

        self.assertLess(_run("couloir", "couloir"), _run("lru", "fifo"))

    def test_fanout_tight_cache_emits_ops(self) -> None:
        wf = create_workflow("fanout", workflow_id="f0")
        engine = SimulationEngine(
            num_workers=2,
            policy_func=get_policy("couloir"),
            policy_name="couloir",
            cache_capacity=40,
            cache_prefix_tokens=20,
        )
        res = engine.run([wf])
        self.assertEqual(res.metrics.num_workflows, 1)
        self.assertGreater(res.metrics.jct_p50, 0.0)
        cache_starts = [e for e in res.event_log if e.get("cache")]
        self.assertGreaterEqual(len(cache_starts), 1)


if __name__ == "__main__":
    unittest.main()
