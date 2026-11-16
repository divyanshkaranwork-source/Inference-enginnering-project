"""PagedAttention memory management (vLLM's core idea) + automatic prefix caching.

The KV cache is one big pool of fixed-size *blocks* (e.g. 16 tokens each).
A sequence owns a *block table*: logical block i -> physical block id.
Like virtual memory pages, blocks need not be contiguous, so:

* no memory is reserved for tokens that have not been generated yet
  (contiguous caches must reserve max_len per request -> 60-80% waste),
* fragmentation is at most block_size - 1 tokens per sequence,
* identical prefixes can *share* physical blocks (reference counted).

Prefix caching: every *full* block gets a content hash that chains the hash
of the previous block - so a hash identifies the entire prefix up to that
block. A new request whose first blocks hash to cached entries reuses them
and skips that part of prefill.

Security: the hash is SHA-256, not Python's ``hash()``. vLLM shipped
``hash()`` originally; CVE-2025-25183 showed crafted prompts could collide
and read another user's cached KV. An optional per-tenant ``cache_salt`` is
mixed into the root hash so tenants never share (or time-probe) each
other's prefixes.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np

from inferlab.runtime.sequence import Sequence


def hash_block(parent: bytes | None, tokens, salt: str | None = None) -> bytes:
    h = hashlib.sha256()
    if parent is None:
        h.update(b"root:")
        if salt:
            h.update(hashlib.sha256(salt.encode("utf-8")).digest())
    else:
        h.update(parent)
    h.update(np.asarray(tokens, dtype="<i4").tobytes())
    return h.digest()


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.ref_count = [0] * num_blocks
        self.block_hash: list[bytes | None] = [None] * num_blocks
        # Free blocks in LRU order (front = evict first). Cached blocks whose
        # ref count dropped to 0 stay here *with* their hash until reused.
        self.free_queue: OrderedDict[int, None] = OrderedDict((i, None) for i in range(num_blocks))
        self.cached: dict[bytes, int] = {}
        # statistics
        self.prefix_queries = 0  # prompt tokens looked up
        self.prefix_hits = 0  # prompt tokens served from cache
        self.evictions = 0

    # ------------------------------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        return len(self.free_queue)

    def usage(self) -> float:
        return 1.0 - self.num_free_blocks / self.num_blocks

    def blocks_needed(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    def can_ever_fit(self, num_tokens: int) -> bool:
        return self.blocks_needed(num_tokens) <= self.num_blocks

    # ------------------------------------------------------------------
    def get_computed_blocks(self, seq: Sequence) -> tuple[list[int], int]:
        """Longest cached prefix of ``seq`` made of full blocks.

        At least one token is always left to compute, because the model must
        run on the last prompt token to produce the first output logits.
        """
        if not self.enable_prefix_caching:
            return [], 0
        tokens = seq.all_tokens
        bs = self.block_size
        max_blocks = (len(tokens) - 1) // bs
        blocks: list[int] = []
        parent = None
        for i in range(max_blocks):
            h = hash_block(parent, tokens[i * bs : (i + 1) * bs], seq.cache_salt)
            bid = self.cached.get(h)
            if bid is None:
                break
            blocks.append(bid)
            parent = h
        return blocks, len(blocks) * bs

    def record_prefix_lookup(self, num_prompt_tokens: int, num_hit_tokens: int) -> None:
        self.prefix_queries += num_prompt_tokens
        self.prefix_hits += num_hit_tokens

    def allocate_slots(
        self, seq: Sequence, num_new_tokens: int, computed_blocks: list[int] | None = None
    ) -> bool:
        """Make sure ``seq`` has blocks for ``num_new_tokens`` more tokens.

        Returns False (and changes nothing) if the pool is too full - the
        scheduler then preempts or waits.
        """
        computed_blocks = computed_blocks or []
        num_computed = seq.num_computed_tokens + len(computed_blocks) * self.block_size
        have = len(seq.block_table) + len(computed_blocks)
        need = self.blocks_needed(num_computed + num_new_tokens) - have
        # computed blocks that are currently free (ref 0) also consume free slots
        reuse_from_free = sum(1 for b in computed_blocks if self.ref_count[b] == 0)
        if max(need, 0) + reuse_from_free > self.num_free_blocks:
            return False
        for b in computed_blocks:  # share cached blocks
            if self.ref_count[b] == 0:
                del self.free_queue[b]
            self.ref_count[b] += 1
            seq.block_table.append(b)
            seq.block_hashes.append(self.block_hash[b])  # type: ignore[arg-type]
        for _ in range(max(need, 0)):
            seq.block_table.append(self._pop_free())
        return True

    def _pop_free(self) -> int:
        bid, _ = self.free_queue.popitem(last=False)
        h = self.block_hash[bid]
        if h is not None:  # evict the cached content of this block
            if self.cached.get(h) == bid:
                del self.cached[h]
            self.block_hash[bid] = None
            self.evictions += 1
        self.ref_count[bid] = 1
        return bid

    def cache_full_blocks(self, seq: Sequence) -> None:
        """Register hashes for blocks that became full after this step."""
        if not self.enable_prefix_caching:
            return
        bs = self.block_size
        tokens = seq.all_tokens
        n_full = seq.num_computed_tokens // bs
        while len(seq.block_hashes) < n_full:
            i = len(seq.block_hashes)
            parent = seq.block_hashes[-1] if seq.block_hashes else None
            h = hash_block(parent, tokens[i * bs : (i + 1) * bs], seq.cache_salt)
            seq.block_hashes.append(h)
            bid = seq.block_table[i]
            if h not in self.cached:
                self.cached[h] = bid
                self.block_hash[bid] = h

    def free(self, seq: Sequence) -> None:
        """Release a sequence's blocks. Tail blocks are freed first so they are
        evicted before the (more shareable) head of the prefix."""
        for bid in reversed(seq.block_table):
            self.ref_count[bid] -= 1
            if self.ref_count[bid] < 0:  # defensive: indicates a bookkeeping bug
                raise RuntimeError(f"block {bid} freed twice")
            if self.ref_count[bid] == 0:
                self.free_queue[bid] = None
        seq.block_table = []
        seq.block_hashes = []

    def reset_prefix_cache(self) -> None:
        for h, bid in list(self.cached.items()):
            if self.ref_count[bid] == 0:
                self.block_hash[bid] = None
                del self.cached[h]

    # ------------------------------------------------------------------
    def snapshot(self, owners: dict[int, list[str]]) -> list[dict]:
        """Per-block state for the UI's KV-cache grid."""
        out = []
        for b in range(self.num_blocks):
            if self.ref_count[b] > 0:
                state = "shared" if self.ref_count[b] > 1 else "used"
            elif self.block_hash[b] is not None:
                state = "cached"
            else:
                state = "free"
            out.append({"id": b, "state": state, "ref": self.ref_count[b], "owners": owners.get(b, [])})
        return out

    def stats(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "free_blocks": self.num_free_blocks,
            "usage": self.usage(),
            "cached_blocks": len(self.cached),
            "prefix_hit_rate": self.prefix_hits / self.prefix_queries if self.prefix_queries else 0.0,
            "prefix_hits_tokens": self.prefix_hits,
            "evictions": self.evictions,
        }
