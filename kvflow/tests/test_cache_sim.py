"""Unit tests for the prefix-cache simulator."""

from __future__ import annotations

import unittest

from kvflow.cache_sim import (
    TRANSFER_TOKENS_PER_SEC,
    NodeStatus,
    PrefixCacheSim,
)
from kvflow.step_graph import steps_to_execution
from kvflow.workflows import sequential_agents_graph


class TestSTERetainsNextAgent(unittest.TestCase):
    def test_ste_keeps_upcoming_prefix_lru_may_evict(self) -> None:
        agents = ["a0", "a1", "a2"]
        shared, fixed = 10, 100
        # Fits shared + two fixed leaves only.
        capacity = shared + 2 * fixed
        ste_map = {"a0": 0, "a1": 1, "a2": 2}

        def primed(policy: str) -> PrefixCacheSim:
            sim = PrefixCacheSim(
                agents=agents,
                shared_tokens=shared,
                fixed_tokens=fixed,
                capacity_tokens=capacity,
                policy=policy,  # type: ignore[arg-type]
                agent_ste=ste_map,
            )
            # Materialize all so CPU backups exist, then pin shared+a0+a2 on GPU.
            for agent in agents:
                sim.access_agent_fixed(agent)
            for node in sim._all_nodes:
                node.status = NodeStatus.CPU
            for node_id in ("shared_root", "fixed:a0", "fixed:a2"):
                node = next(n for n in sim._all_nodes if n.node_id == node_id)
                node.status = NodeStatus.GPU
            sim.fixed["a0"].last_access = 1.0
            sim.fixed["a2"].last_access = 10.0  # more recent → LRU keeps a2
            sim.root.last_access = 10.0
            sim.update_ste(ste_map)
            return sim

        # Make room for next agent a1. Protect only shared + a1 so both a0 and a2
        # are eviction candidates — STE should drop far-future a2; LRU drops older a0.
        protect = {"shared_root", "fixed:a1"}

        ste_sim = primed("ste")
        ste_sim._evict_until_fit(fixed, protect=protect)
        self.assertEqual(ste_sim.fixed["a0"].status, NodeStatus.GPU)
        self.assertEqual(ste_sim.fixed["a2"].status, NodeStatus.CPU)

        lru_sim = primed("lru")
        lru_sim._evict_until_fit(fixed, protect=protect)
        self.assertEqual(lru_sim.fixed["a0"].status, NodeStatus.CPU)
        self.assertEqual(lru_sim.fixed["a2"].status, NodeStatus.GPU)

        ste_sim.access_agent_fixed("a1")
        self.assertEqual(ste_sim.fixed["a1"].status, NodeStatus.GPU)
        self.assertEqual(ste_sim.fixed["a0"].status, NodeStatus.GPU)

        lru_sim.access_agent_fixed("a1")
        self.assertEqual(lru_sim.fixed["a1"].status, NodeStatus.GPU)
        # Upcoming/current low-STE a0 was sacrificed by LRU.
        self.assertEqual(lru_sim.fixed["a0"].status, NodeStatus.CPU)


class TestSharedRootSurvives(unittest.TestCase):
    def test_shared_root_kept_when_child_has_low_ste(self) -> None:
        agents = ["hot", "cold"]
        shared, fixed = 50, 100
        capacity = shared + fixed  # only one fixed leaf + shared

        sim = PrefixCacheSim(
            agents=agents,
            shared_tokens=shared,
            fixed_tokens=fixed,
            capacity_tokens=capacity,
            policy="ste",
            agent_ste={"hot": 0, "cold": 5},
        )
        sim.access_agent_fixed("hot")
        sim.access_agent_fixed("cold")
        # cold's fixed may replace hot's fixed, but shared root has min STE of children
        # and should stay — it is on the path for cold and protected during access.
        self.assertEqual(sim.root.status, NodeStatus.GPU)

        # Evict under pressure from a fresh dynamic-sized demand: access hot again.
        sim.access_agent_fixed("hot")
        self.assertEqual(sim.root.status, NodeStatus.GPU)
        # cold leaf should be preferred for eviction (higher STE).
        self.assertEqual(sim.fixed["cold"].status, NodeStatus.CPU)


class TestPrefetchReducesStall(unittest.TestCase):
    def test_prefetch_reduces_stall_when_decode_covers_transfer(self) -> None:
        agents = ["a0", "a1"]
        shared, fixed = 10, 1000
        capacity = shared + 2 * fixed
        # Transfer time for fixed leaf alone.
        transfer_s = fixed / TRANSFER_TOKENS_PER_SEC
        # Decode long enough to overlap the transfer.
        output_tokens = int(transfer_s * 80) + 50  # DECODE_TOKENS_PER_SEC = 80

        def run(policy: str) -> float:
            sim = PrefixCacheSim(
                agents=agents,
                shared_tokens=shared,
                fixed_tokens=fixed,
                capacity_tokens=capacity,
                policy=policy,  # type: ignore[arg-type]
                agent_ste={"a0": 0, "a1": 1},
            )
            # Materialize both once so CPU backups exist.
            sim.access_agent_fixed("a0")
            sim.access_agent_fixed("a1")
            # Evict a1 to CPU.
            sim.fixed["a1"].status = NodeStatus.CPU
            sim.root.status = NodeStatus.GPU
            sim.fixed["a0"].status = NodeStatus.GPU

            sim.update_ste({"a0": 0, "a1": 1})
            sim.start_prefetch(["a1"])
            sim.advance_decode(output_tokens)
            before = sim.stats.stall_s
            sim.access_agent_fixed("a1")
            return sim.stats.stall_s - before

        stall_ste = run("ste")
        stall_prefetch = run("ste+prefetch")
        self.assertGreater(stall_ste, 0.0)
        self.assertLess(stall_prefetch, stall_ste)
        self.assertAlmostEqual(stall_prefetch, 0.0, places=6)


class TestDynamicAlwaysEvictable(unittest.TestCase):
    def test_dynamic_evicted_before_fixed(self) -> None:
        agents = ["a0", "a1"]
        shared, fixed, dyn = 10, 100, 40
        capacity = shared + fixed + dyn  # one agent fully; second forces eviction

        sim = PrefixCacheSim(
            agents=agents,
            shared_tokens=shared,
            fixed_tokens=fixed,
            capacity_tokens=capacity,
            policy="ste",
            agent_ste={"a0": 0, "a1": 1},
        )
        sim.access_agent("a0", dynamic_tokens=dyn)
        self.assertEqual(sim.dynamics["a0"].status, NodeStatus.GPU)
        sim.access_agent_fixed("a1")
        # Dynamic of a0 should be gone; a0 fixed may or may not remain depending on room.
        self.assertEqual(sim.dynamics["a0"].status, NodeStatus.CPU)


class TestSequentialSTEIntegration(unittest.TestCase):
    def test_workflow_ste_updates(self) -> None:
        graph = sequential_agents_graph(3)
        ste = steps_to_execution(graph, active_agents=["agent_0"])
        self.assertEqual(ste["agent_0"], 0)
        self.assertEqual(ste["agent_1"], 1)
        self.assertEqual(ste["agent_2"], 2)


if __name__ == "__main__":
    unittest.main()
