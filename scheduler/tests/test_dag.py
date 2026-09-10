"""Unit tests for WorkflowDAG and Task data structures."""

from __future__ import annotations

import unittest
from pathlib import Path

from scheduler.dag import (
    LLM_PROFILE_MEANS,
    SIMULATED_PROFILE_MEANS,
    WorkflowDAG,
    create_research_writing_workflow,
    load_profile_means,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LLM_CSV = _REPO_ROOT / "llm" / "profiling_summary_llm.csv"
_SIM_CSV = _REPO_ROOT / "simulated" / "profiling_summary.csv"

# Known table values (2 d.p.) used throughout the scheduling report.
_EXPECTED_LLM_MEANS = {
    "planner": 21.40,
    "researcher_a": 10.85,
    "researcher_b": 41.47,
    "researcher_c": 19.72,
    "synthesizer": 21.49,
    "critic": 13.61,
    "final_writer": 20.78,
}
_EXPECTED_SIM_MEANS = {
    "planner": 0.77,
    "researcher_a": 1.52,
    "researcher_b": 3.08,
    "researcher_c": 1.60,
    "synthesizer": 0.88,
    "critic": 0.97,
    "final_writer": 0.96,
}


class TestProfileMeansLoader(unittest.TestCase):
    def test_load_llm_profile_means_from_csv(self) -> None:
        self.assertTrue(_LLM_CSV.is_file(), f"Missing {_LLM_CSV}")
        means = load_profile_means(_LLM_CSV)
        self.assertEqual(set(means), set(_EXPECTED_LLM_MEANS))
        for agent, expected in _EXPECTED_LLM_MEANS.items():
            self.assertAlmostEqual(means[agent], expected, places=2)

    def test_load_simulated_profile_means_from_csv(self) -> None:
        self.assertTrue(_SIM_CSV.is_file(), f"Missing {_SIM_CSV}")
        means = load_profile_means(_SIM_CSV)
        self.assertEqual(set(means), set(_EXPECTED_SIM_MEANS))
        for agent, expected in _EXPECTED_SIM_MEANS.items():
            self.assertAlmostEqual(means[agent], expected, places=2)

    def test_module_means_match_csv(self) -> None:
        for agent, expected in _EXPECTED_LLM_MEANS.items():
            self.assertAlmostEqual(LLM_PROFILE_MEANS[agent], expected, places=2)
        for agent, expected in _EXPECTED_SIM_MEANS.items():
            self.assertAlmostEqual(SIMULATED_PROFILE_MEANS[agent], expected, places=2)


class TestWorkflowDAG(unittest.TestCase):
    def test_workflow_creation(self) -> None:
        wf = create_research_writing_workflow(workflow_id="wf0", profile="llm")
        self.assertEqual(len(wf.tasks), 7)
        self.assertEqual(len(wf.edges), 8)
        self.assertEqual(len(wf.predecessors["wf0_planner"]), 0)
        self.assertEqual(len(wf.successors["wf0_planner"]), 3)
        self.assertEqual(len(wf.predecessors["wf0_synthesizer"]), 3)

    def test_upward_ranks_and_critical_path(self) -> None:
        wf = create_research_writing_workflow(workflow_id="wf0", profile="llm")
        ranks = wf.compute_upward_ranks(use_expected=True)

        # Expected downstream:
        # final_writer: 20.78
        # critic: 13.61 + 20.78 = 34.39
        # synthesizer: 21.49 + 34.39 = 55.88
        # researcher_b (41.47): 41.47 + 55.88 = 97.35
        # researcher_a (10.85): 10.85 + 55.88 = 66.73
        # planner (21.40): 21.40 + max(66.73, 97.35, 75.60) = 118.75
        self.assertAlmostEqual(ranks["wf0_final_writer"], 20.78, places=2)
        self.assertAlmostEqual(ranks["wf0_critic"], 34.39, places=2)
        self.assertAlmostEqual(ranks["wf0_synthesizer"], 55.88, places=2)
        self.assertAlmostEqual(ranks["wf0_researcher_b"], 97.35, places=2)
        self.assertAlmostEqual(ranks["wf0_researcher_a"], 66.73, places=2)
        self.assertAlmostEqual(ranks["wf0_planner"], 118.75, places=2)

        cpath, cpath_len = wf.compute_critical_path(use_expected=True)
        expected_path = [
            "wf0_planner",
            "wf0_researcher_b",
            "wf0_synthesizer",
            "wf0_critic",
            "wf0_final_writer",
        ]
        self.assertEqual(cpath, expected_path)
        self.assertAlmostEqual(cpath_len, 118.75, places=2)

    def test_static_slack(self) -> None:
        wf = create_research_writing_workflow(workflow_id="wf0", profile="llm")
        slacks = wf.compute_static_slacks(use_expected=True)

        # Critical path nodes must have 0 slack
        self.assertAlmostEqual(slacks["wf0_planner"], 0.0, places=2)
        self.assertAlmostEqual(slacks["wf0_researcher_b"], 0.0, places=2)
        self.assertAlmostEqual(slacks["wf0_synthesizer"], 0.0, places=2)

        # Non-critical researchers must have positive slack
        # researcher_b (41.47) vs researcher_a (10.85) -> slack = 41.47 - 10.85 = 30.62
        self.assertAlmostEqual(slacks["wf0_researcher_a"], 30.62, places=2)
        # researcher_c (19.72) -> slack = 41.47 - 19.72 = 21.75
        self.assertAlmostEqual(slacks["wf0_researcher_c"], 21.75, places=2)


if __name__ == "__main__":
    unittest.main()
