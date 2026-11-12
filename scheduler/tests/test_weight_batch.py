"""Tests for weight ablation hooks and random/batch DAG generators."""

from __future__ import annotations

import unittest

from scheduler.batch_couloir import run_batch_matrix, run_random_single_family
from scheduler.couloir import CouloirWeights
from scheduler.engine import SimulationEngine
from scheduler.policies import get_policy
from scheduler.random_dags import create_random_layered_dag
from scheduler.weight_ablation import run_leave_one_out
from scheduler.workloads import create_workflow


class TestWeightsAndBatch(unittest.TestCase):
    def test_engine_accepts_weights(self) -> None:
        wf = create_workflow("fanout", workflow_id="w0")
        w = CouloirWeights(cp=2.0, slack=0.0, reuse=1.0, evict=0.0, age=0.0)
        eng = SimulationEngine(
            num_workers=2,
            policy_func=get_policy("couloir"),
            policy_name="couloir",
            cache_capacity=2048,
            cache_prefix_tokens=512,
            weights=w,
        )
        res = eng.run([wf])
        self.assertGreater(res.metrics.makespan, 0.0)

    def test_leave_one_out_has_rows(self) -> None:
        df = run_leave_one_out(dags=("fanout",), k=2, c=1024, prefix=512)
        self.assertIn("dropped_term", df.columns)
        self.assertGreater(len(df), 5)

    def test_random_dag_acyclic_runnable(self) -> None:
        wf = create_random_layered_dag("r0", n_nodes=8, n_layers=4)
        self.assertEqual(len(wf.tasks), 8)
        eng = SimulationEngine(
            num_workers=2,
            policy_func=get_policy("cpf"),
            policy_name="cpf",
        )
        res = eng.run([wf])
        self.assertGreater(res.metrics.makespan, 0.0)

    def test_batch_and_solo_smoke(self) -> None:
        batch = run_batch_matrix(
            shapes=("random",),
            batch_sizes=(2,),
            num_workers=2,
            aging_modes=(True,),
            seed=1,
        )
        self.assertGreater(len(batch), 0)
        solo = run_random_single_family(n_dags=2, num_workers=2, seed=1)
        self.assertEqual(len(solo), 2 * 4)


if __name__ == "__main__":
    unittest.main()
