"""SGLang's RadixAttention prefix cache - a radix tree over token sequences.

vLLM's prefix cache (``block_manager.py``) hashes *full blocks*. SGLang
instead keeps a radix tree whose edges are token runs and whose nodes point
at KV-cache slots. Benefits:

* token-level granularity (no block-size rounding),
* natural sharing for branching workloads: few-shot prompts, multi-turn chat,
  tree-of-thought, agents re-sending a growing transcript,
* LRU eviction of *leaves* only, so shared prefixes survive longest,
* "cache-aware scheduling": run waiting requests with the longest matched
  prefix first to maximise hit rate.

This module is a self-contained, faithful re-implementation of the data
structure; the UI's Runtime page visualises it.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field


@dataclass(eq=False)
class RadixNode:
    key: list[int] = field(default_factory=list)  # token run on the edge into this node
    children: dict[int, RadixNode] = field(default_factory=dict)
    parent: RadixNode | None = None
    lock_ref: int = 0  # >0 while a running request uses this prefix
    last_access: float = field(default_factory=time.monotonic)
    node_id: int = 0


class RadixCache:
    def __init__(self, capacity_tokens: int = 4096):
        if capacity_tokens <= 0:
            raise ValueError("capacity must be positive")
        self._ids = itertools.count()
        self.root = RadixNode(node_id=next(self._ids))
        self.capacity = capacity_tokens
        self.total_tokens = 0
        self.hits = 0
        self.queries = 0
        self.evicted_tokens = 0

    # ------------------------------------------------------------------
    def match_prefix(self, tokens: list[int]) -> tuple[int, RadixNode]:
        """Length of the longest cached prefix and the deepest node reached."""
        node, i = self.root, 0
        now = time.monotonic()
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                break
            k = child.key
            n = 0
            while n < len(k) and i + n < len(tokens) and k[n] == tokens[i + n]:
                n += 1
            child.last_access = now
            i += n
            if n < len(k):
                break  # partial edge match
            node = child
        return i, node

    def _split(self, child: RadixNode, n: int) -> RadixNode:
        """Split ``child``'s edge after n tokens; returns the new middle node."""
        mid = RadixNode(key=child.key[:n], parent=child.parent, node_id=next(self._ids))
        mid.lock_ref = child.lock_ref
        mid.last_access = child.last_access
        if child.parent is None:  # the root is never split
            raise RuntimeError("cannot split the root node")
        child.parent.children[child.key[0]] = mid
        child.key = child.key[n:]
        child.parent = mid
        mid.children[child.key[0]] = child
        return mid

    def insert(self, tokens: list[int]) -> int:
        """Insert a sequence; returns how many of its tokens were already cached."""
        self.queries += len(tokens)
        node, i, matched = self.root, 0, 0
        now = time.monotonic()
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                new = RadixNode(key=list(tokens[i:]), parent=node, node_id=next(self._ids))
                node.children[tokens[i]] = new
                self.total_tokens += len(tokens) - i
                break
            k = child.key
            n = 0
            while n < len(k) and i + n < len(tokens) and k[n] == tokens[i + n]:
                n += 1
            child.last_access = now
            if n < len(k):
                child = self._split(child, n)
            node = child
            i += n
            matched += n
        self.hits += matched
        self._evict_if_needed()
        return matched

    # ------------------------------------------------------------------
    def _leaves(self) -> list[RadixNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if not n.children and n is not self.root:
                out.append(n)
            stack.extend(n.children.values())
        return out

    def _evict_if_needed(self) -> None:
        while self.total_tokens > self.capacity:
            leaves = [n for n in self._leaves() if n.lock_ref == 0]
            if not leaves:
                return
            victim = min(leaves, key=lambda n: n.last_access)  # LRU leaf
            self.evict_node(victim)

    def evict_node(self, node: RadixNode) -> None:
        parent = node.parent
        if parent is None:
            return
        del parent.children[node.key[0]]
        self.total_tokens -= len(node.key)
        self.evicted_tokens += len(node.key)

    def lock(self, node: RadixNode, delta: int) -> None:
        while node is not None and node is not self.root:
            node.lock_ref += delta
            node = node.parent  # type: ignore[assignment]

    # ------------------------------------------------------------------
    def to_dict(self, decode=None, max_nodes: int = 200) -> dict:
        """Tree for the UI; ``decode`` turns a token list into display text."""
        count = 0

        def walk(n: RadixNode) -> dict:
            nonlocal count
            count += 1
            children = []
            for c in sorted(n.children.values(), key=lambda c: c.key[0]):
                if count >= max_nodes:
                    break
                children.append(walk(c))
            return {
                "id": n.node_id,
                "tokens": len(n.key),
                "text": decode(n.key) if decode and n.key else "",
                "children": children,
            }

        return {
            "tree": walk(self.root),
            "total_tokens": self.total_tokens,
            "capacity": self.capacity,
            "hit_rate": self.hits / self.queries if self.queries else 0.0,
            "evicted_tokens": self.evicted_tokens,
        }
