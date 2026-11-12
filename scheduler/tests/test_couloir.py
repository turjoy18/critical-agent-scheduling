"""Unit tests for Couloir utility scoring and aging."""

from __future__ import annotations

import unittest

from scheduler.couloir import (
    CacheView,
    CouloirWeights,
    decide,
    remaining_critical_path,
    score_ready_tasks,
)
from scheduler.dag import create_research_writing_workflow


class TestCouloirUtility(unittest.TestCase):
    def setUp(self) -> None:
        self.wf = create_research_writing_workflow(workflow_id="wf0", profile="llm")
        self.workflows = {"wf0": self.wf}
        self.r_a = self.wf.tasks["wf0_researcher_a"]
        self.r_b = self.wf.tasks["wf0_researcher_b"]
        self.r_c = self.wf.tasks["wf0_researcher_c"]
        for t in (self.r_a, self.r_b, self.r_c):
            t.ready_time = 21.40

    def test_remaining_cp_matches_upward_rank(self) -> None:
        ranks = self.wf.compute_upward_ranks(use_expected=True)
        for t in (self.r_a, self.r_b, self.r_c):
            self.assertAlmostEqual(
                remaining_critical_path(t, self.workflows), ranks[t.task_id], places=6
            )

    def test_without_cache_prefers_researcher_b(self) -> None:
        ready = [self.r_a, self.r_b, self.r_c]
        decision = decide(ready, 21.40, self.workflows, enable_aging=True)
        self.assertEqual(decision.ordered_tasks[0].agent_type, "researcher_b")
        self.assertEqual(decision.selected_task_id, self.r_b.task_id)

    def test_deterministic_scores(self) -> None:
        ready = [self.r_c, self.r_a, self.r_b]
        a = score_ready_tasks(ready, 21.40, self.workflows)
        b = score_ready_tasks(ready, 21.40, self.workflows)
        self.assertEqual(
            [(s.task_id, round(s.utility, 10)) for s in a],
            [(s.task_id, round(s.utility, 10)) for s in b],
        )

    def test_reuse_benefit_boosts_cached_agent(self) -> None:
        cache = CacheView(
            capacity_tokens=100,
            gpu_tokens=40,
            agent_on_gpu={"researcher_a": True, "researcher_b": False, "researcher_c": False},
            agent_prefix_tokens={
                "researcher_a": 40,
                "researcher_b": 40,
                "researcher_c": 40,
            },
        )
        # Zero aging/cp emphasis so reuse can dominate a near-tie on A vs C.
        weights = CouloirWeights(cp=0.1, slack=0.1, reuse=5.0, evict=0.0, age=0.0)
        ready = [self.r_a, self.r_c]
        decision = decide(
            ready, 21.40, self.workflows, cache=cache, weights=weights, enable_aging=False
        )
        self.assertEqual(decision.ordered_tasks[0].agent_type, "researcher_a")

    def test_aging_prevents_starvation(self) -> None:
        """Long-waiting short task eventually outranks a fresh critical task."""
        self.r_a.ready_time = 0.0
        self.r_b.ready_time = 100.0
        ready = [self.r_a, self.r_b]
        weights = CouloirWeights(cp=1.0, slack=0.0, reuse=0.0, evict=0.0, age=3.0)

        no_age = decide(
            ready, 100.0, self.workflows, weights=weights, enable_aging=False
        )
        with_age = decide(
            ready, 100.0, self.workflows, weights=weights, enable_aging=True
        )

        # Without aging, B (higher CP) wins; with aging, A wins.
        self.assertEqual(no_age.ordered_tasks[0].agent_type, "researcher_b")
        self.assertEqual(with_age.ordered_tasks[0].agent_type, "researcher_a")

    def test_cache_action_keeps_selected_and_lists_evictions(self) -> None:
        cache = CacheView(
            capacity_tokens=80,
            gpu_tokens=80,
            agent_on_gpu={
                "researcher_a": True,
                "researcher_b": True,
                "researcher_c": True,
                "planner": True,
            },
            agent_prefix_tokens={
                "researcher_a": 20,
                "researcher_b": 20,
                "researcher_c": 20,
                "planner": 20,
            },
        )
        ready = [self.r_a, self.r_b, self.r_c]
        decision = decide(ready, 21.40, self.workflows, cache=cache, prefetch_top_k=1)
        self.assertIn(decision.ordered_tasks[0].agent_type, decision.cache_action.keep_agents)
        self.assertTrue(
            any(a not in decision.cache_action.keep_agents for a in decision.cache_action.evict_agents)
            or decision.cache_action.evict_agents == []
            or "planner" in decision.cache_action.evict_agents
        )


if __name__ == "__main__":
    unittest.main()
