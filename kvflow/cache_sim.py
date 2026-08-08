"""Pure-Python prefix-cache simulator (KVFlow-inspired eviction + statuses)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

PolicyName = Literal["lru", "ste", "ste+prefetch"]


class NodeStatus(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    LOADING = "loading"


# Synthetic timing model (tokens / rate). Chosen for reproducibility, not hardware parity.
RECOMPUTE_TOKENS_PER_SEC = 8_000.0
TRANSFER_TOKENS_PER_SEC = 40_000.0
DECODE_TOKENS_PER_SEC = 80.0


@dataclass
class CacheNode:
    node_id: str
    tokens: int
    is_dynamic: bool = False
    agent: str | None = None
    parent: CacheNode | None = None
    children: list[CacheNode] = field(default_factory=list)
    status: NodeStatus = NodeStatus.CPU
    last_access: float = -1.0
    ste_priority: float = 0.0
    load_ready_at: float = 0.0

    def path_to_root(self) -> list[CacheNode]:
        nodes: list[CacheNode] = []
        cur: CacheNode | None = self
        while cur is not None:
            nodes.append(cur)
            cur = cur.parent
        nodes.reverse()
        return nodes


@dataclass
class AccessResult:
    hit: bool
    stall_s: float
    recompute_tokens: int
    transfer_tokens: int
    time_s: float


@dataclass
class SimStats:
    hits: int = 0
    misses: int = 0
    recompute_tokens: int = 0
    transfer_tokens: int = 0
    stall_s: float = 0.0
    eviction_count: int = 0
    workflow_latency_s: float = 0.0

    @property
    def accesses(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0


class PrefixCacheSim:
    """Radix-like tree: shared root + per-agent fixed leaf (+ optional dynamic)."""

    def __init__(
        self,
        agents: list[str],
        shared_tokens: int,
        fixed_tokens: int,
        capacity_tokens: int,
        policy: PolicyName = "lru",
        agent_ste: dict[str, int] | None = None,
    ) -> None:
        if capacity_tokens < 1:
            raise ValueError("capacity_tokens must be >= 1")
        self.agents = list(agents)
        self.capacity_tokens = capacity_tokens
        self.policy: PolicyName = policy
        self.agent_ste = dict(agent_ste or {a: i for i, a in enumerate(agents)})
        self.now = 0.0
        self.stats = SimStats()

        self.root = CacheNode(node_id="shared_root", tokens=shared_tokens, agent=None)
        self.fixed: dict[str, CacheNode] = {}
        self.dynamics: dict[str, CacheNode] = {}
        for agent in agents:
            leaf = CacheNode(
                node_id=f"fixed:{agent}",
                tokens=fixed_tokens,
                agent=agent,
                parent=self.root,
            )
            self.root.children.append(leaf)
            self.fixed[agent] = leaf

        self._all_nodes: list[CacheNode] = [self.root, *self.fixed.values()]
        self._refresh_ste_priorities()

    def _refresh_ste_priorities(self) -> None:
        for agent, leaf in self.fixed.items():
            leaf.ste_priority = float(self.agent_ste.get(agent, 0))
        # Shared ancestors keep the minimum (least evictable) child priority.
        if self.root.children:
            self.root.ste_priority = min(c.ste_priority for c in self.root.children)
        else:
            self.root.ste_priority = 0.0
        for dyn in self.dynamics.values():
            # Dynamics are always preferred eviction candidates.
            dyn.ste_priority = float("inf")

    def update_ste(self, agent_ste: dict[str, int]) -> None:
        self.agent_ste = dict(agent_ste)
        self._refresh_ste_priorities()

    def gpu_tokens(self) -> int:
        return sum(n.tokens for n in self._all_nodes if n.status == NodeStatus.GPU)

    def _touch(self, node: CacheNode) -> None:
        node.last_access = self.now

    def _eviction_score(self, node: CacheNode) -> tuple:
        """Higher score = more evictable."""
        if node.is_dynamic:
            return (2, 0.0, -node.last_access)
        if self.policy == "lru":
            return (1, -node.last_access, 0.0)
        # ste / ste+prefetch: larger STE first
        return (1, node.ste_priority, -node.last_access)

    def _evict_until_fit(self, extra_tokens: int, protect: set[str]) -> None:
        need = self.gpu_tokens() + extra_tokens - self.capacity_tokens
        if need <= 0:
            return
        candidates = [
            n
            for n in self._all_nodes
            if n.status == NodeStatus.GPU and n.node_id not in protect
        ]
        candidates.sort(key=self._eviction_score, reverse=True)
        freed = 0
        for node in candidates:
            if freed >= need:
                break
            node.status = NodeStatus.CPU
            freed += node.tokens
            self.stats.eviction_count += 1

    def _ensure_gpu(self, node: CacheNode, protect: set[str]) -> AccessResult:
        """Bring ``node`` onto GPU, accounting for load/recompute and capacity."""
        stall = 0.0

        if node.status == NodeStatus.GPU:
            self._touch(node)
            return AccessResult(True, 0.0, 0, 0, 0.0)

        if node.status == NodeStatus.LOADING:
            if self.now < node.load_ready_at:
                stall = node.load_ready_at - self.now
                self.now = node.load_ready_at
            node.status = NodeStatus.GPU
            self._touch(node)
            return AccessResult(True, stall, 0, 0, stall)

        protect = set(protect) | {node.node_id}
        # Never protect dynamics of other agents when bringing a node in.
        self._evict_until_fit(node.tokens, protect=protect)

        if self.gpu_tokens() + node.tokens > self.capacity_tokens:
            self._evict_until_fit(node.tokens, protect={node.node_id})

        if node.last_access < 0:
            recompute = node.tokens
            cost = recompute / RECOMPUTE_TOKENS_PER_SEC
            self.now += cost
            node.status = NodeStatus.GPU
            self._touch(node)
            # On-demand recompute blocks the agent (counts as stall).
            return AccessResult(False, cost, recompute, 0, cost)

        transfer = node.tokens
        cost = transfer / TRANSFER_TOKENS_PER_SEC
        self.now += cost
        node.status = NodeStatus.GPU
        self._touch(node)
        # On-demand CPU→GPU load blocks the agent (counts as stall).
        return AccessResult(False, cost, 0, transfer, cost)

    def access_agent_fixed(self, agent: str) -> AccessResult:
        """Ensure shared root + agent fixed leaf are on GPU."""
        return self.access_agent(agent, dynamic_tokens=0)

    def _get_or_create_dynamic(self, agent: str, tokens: int) -> CacheNode:
        if tokens < 1:
            raise ValueError("tokens must be >= 1")
        leaf = self.fixed[agent]
        dyn = self.dynamics.get(agent)
        if dyn is not None and dyn.tokens == tokens:
            return dyn
        if dyn is not None:
            if dyn in leaf.children:
                leaf.children.remove(dyn)
            if dyn in self._all_nodes:
                self._all_nodes.remove(dyn)
        dyn = CacheNode(
            node_id=f"dynamic:{agent}",
            tokens=tokens,
            is_dynamic=True,
            agent=agent,
            parent=leaf,
            ste_priority=float("inf"),
        )
        leaf.children.append(dyn)
        self.dynamics[agent] = dyn
        self._all_nodes.append(dyn)
        return dyn

    def access_agent(self, agent: str, dynamic_tokens: int = 0) -> AccessResult:
        """One logical access: shared+fixed path, optionally plus dynamic suffix.

        Hit/miss is decided by the agent's **fixed** leaf: already on GPU or
        LOADING (retained / prefetched) counts as a hit even if shared root or
        dynamic suffix still needs work.
        """
        if agent not in self.fixed:
            raise KeyError(agent)
        leaf = self.fixed[agent]
        fixed_was_ready = leaf.status in (NodeStatus.GPU, NodeStatus.LOADING)

        path = list(leaf.path_to_root())
        if dynamic_tokens > 0:
            path.append(self._get_or_create_dynamic(agent, dynamic_tokens))

        # Protect fixed/shared path only — dynamics remain evictable.
        protect = {n.node_id for n in leaf.path_to_root()}

        total_stall = 0.0
        total_recompute = 0
        total_transfer = 0
        total_time = 0.0

        for node in path:
            result = self._ensure_gpu(node, protect=protect)
            total_stall += result.stall_s
            total_recompute += result.recompute_tokens
            total_transfer += result.transfer_tokens
            total_time += result.time_s

        if fixed_was_ready:
            self.stats.hits += 1
        else:
            self.stats.misses += 1
        self.stats.stall_s += total_stall
        self.stats.recompute_tokens += total_recompute
        self.stats.transfer_tokens += total_transfer
        return AccessResult(
            fixed_was_ready, total_stall, total_recompute, total_transfer, total_time
        )

    def start_prefetch(self, agents: list[str]) -> None:
        """Begin overlapped CPU→GPU loads for agents' fixed prefixes (and shared root)."""
        if self.policy != "ste+prefetch":
            return
        loading_protect = {
            n.node_id for n in self._all_nodes if n.status == NodeStatus.LOADING
        }
        for agent in agents:
            if agent not in self.fixed:
                continue
            leaf = self.fixed[agent]
            for node in leaf.path_to_root():
                if node.status in (NodeStatus.GPU, NodeStatus.LOADING):
                    continue
                protect = loading_protect | {node.node_id}
                if node.last_access < 0:
                    # Must materialize once via recompute before CPU backup exists.
                    self._ensure_gpu(node, protect={node.node_id})
                    continue
                self._evict_until_fit(node.tokens, protect=protect)
                if self.gpu_tokens() + node.tokens > self.capacity_tokens:
                    continue
                node.status = NodeStatus.LOADING
                node.load_ready_at = self.now + node.tokens / TRANSFER_TOKENS_PER_SEC
                loading_protect.add(node.node_id)

    def complete_due_loads(self) -> None:
        for node in self._all_nodes:
            if node.status == NodeStatus.LOADING and self.now >= node.load_ready_at:
                node.status = NodeStatus.GPU
                self._touch(node)

    def advance_decode(self, output_tokens: int) -> float:
        """Advance simulation clock by decode time for the current agent."""
        dt = output_tokens / DECODE_TOKENS_PER_SEC
        self.now += dt
        self.complete_due_loads()
        return dt

    def advance_time(self, seconds: float) -> None:
        self.now += seconds
        self.complete_due_loads()

    def agents_with_ste(self, ste_value: int) -> list[str]:
        return [a for a, ste in self.agent_ste.items() if ste == ste_value]
