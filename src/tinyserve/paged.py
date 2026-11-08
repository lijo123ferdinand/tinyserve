"""Paged KV cache: block allocator, block tables, copy-on-write, prefix caching,
and paged attention.

The idea (Kwon et al., "Efficient Memory Management for LLM Serving with
PagedAttention", SOSP'23) is the OS virtual-memory trick applied to the KV
cache:

    logical token position p  --block table-->  physical block, offset p % B

    seq A  block table [7, 2, 9]      pool:  [0][1][2][3][4][5][6][7][8][9]...
    seq B  block table [7, 4]                        ^A1   ^B1   ^A0B0  ^A2
                        ^ shared prefix block, ref_count=2

* Memory is allocated one fixed-size block at a time as a sequence grows, so
  waste is bounded by ``block_size - 1`` slots per sequence instead of
  ``max_len - len``.
* Blocks are reference counted, so sequences can *share* physical blocks:
  forks of a parallel-sampling request, and unrelated requests with the same
  prompt prefix (prefix caching). A shared block that must be written is
  copied first (copy-on-write).
* Freed blocks that hold a full, hashed prefix are not wiped - they go to an
  LRU "evictable" set and can be revived by a later request with the same
  prefix. They are reclaimed only when the free list runs dry.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from dataclasses import dataclass, field

import numpy as np

from .model import ModelConfig, causal_attention


class OutOfBlocks(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Block allocator
# --------------------------------------------------------------------------


class BlockAllocator:
    """Physical block bookkeeping: free list, ref counts, prefix-hash cache."""

    def __init__(self, num_blocks: int, enable_prefix_caching: bool = True):
        self.num_blocks = num_blocks
        self.enable_prefix_caching = enable_prefix_caching
        self.free: deque[int] = deque(range(num_blocks))
        self.ref = np.zeros(num_blocks, dtype=np.int64)
        self.block_hash: dict[int, bytes] = {}       # block id -> content hash
        self.cached: dict[bytes, int] = {}            # content hash -> block id
        self.evictable: OrderedDict[int, None] = OrderedDict()  # ref==0 but hashed, LRU order
        # stats
        self.prefix_hits = 0
        self.prefix_queries = 0

    # -- capacity ---------------------------------------------------------
    @property
    def num_free(self) -> int:
        """Blocks obtainable right now (truly free + evictable cached)."""
        return len(self.free) + len(self.evictable)

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_free

    # -- alloc / free -----------------------------------------------------
    def allocate(self) -> int:
        if self.free:
            b = self.free.popleft()
        elif self.evictable:
            # Evict the least-recently-freed cached block. Its content is lost,
            # so it must leave the hash map too.
            b, _ = self.evictable.popitem(last=False)
            del self.cached[self.block_hash.pop(b)]
        else:
            raise OutOfBlocks("KV cache is full")
        assert self.ref[b] == 0
        self.ref[b] = 1
        return b

    def incref(self, b: int) -> None:
        if self.ref[b] == 0:
            # reviving a cached-but-unreferenced block
            self.evictable.pop(b)
        self.ref[b] += 1

    def decref(self, b: int) -> None:
        if self.ref[b] <= 0:
            raise RuntimeError(f"double free of block {b}")
        self.ref[b] -= 1
        if self.ref[b] == 0:
            if b in self.block_hash:
                self.evictable[b] = None  # keep content for future prefix hits
            else:
                self.free.append(b)

    # -- prefix cache -----------------------------------------------------
    def lookup(self, h: bytes) -> int | None:
        return self.cached.get(h) if self.enable_prefix_caching else None

    def register(self, b: int, h: bytes) -> None:
        """Publish a now-full block under its content hash.

        If an identical block is already published (two requests with the same
        prefix prefilled concurrently) we keep the existing entry; the duplicate
        simply stays private and is freed normally.
        """
        if not self.enable_prefix_caching or h in self.cached or b in self.block_hash:
            return
        self.cached[h] = b
        self.block_hash[b] = h


def block_hash(parent: bytes, tokens) -> bytes:
    """Hash of a full block *including everything before it*.

    Chaining in the parent hash makes the key identify the whole prefix, not
    just this block's tokens: "the cat" at position 0 and at position 512 must
    not collide, because their K/V differ (RoPE, and attention to different
    histories). A cryptographic hash avoids the silent-wrong-output failure
    mode of a 64-bit ``hash()`` collision.
    """
    m = hashlib.sha256(parent)
    m.update(np.asarray(tokens, dtype=np.int64).tobytes())
    return m.digest()


# --------------------------------------------------------------------------
# Block manager: per-sequence block tables on top of the allocator
# --------------------------------------------------------------------------


@dataclass
class BlockTable:
    blocks: list[int] = field(default_factory=list)
    # hashes of this sequence's leading full blocks (prefix chain), so we never
    # rehash a block twice
    hashes: list[bytes] = field(default_factory=list)


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        self.block_size = block_size
        self.alloc = BlockAllocator(num_blocks, enable_prefix_caching)
        self.tables: dict[int, BlockTable] = {}

    def blocks_needed(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    # -- admission --------------------------------------------------------
    def plan_prefix(self, tokens) -> tuple[list[int], list[bytes]]:
        """Find cached blocks for the longest full-block prefix of ``tokens``.

        At most ``(len-1) // block_size`` blocks are reused: the last prompt
        token is always recomputed because we need its logits to sample the
        first output token. A side benefit: every write then lands in a fresh,
        private block, so prefix sharing never needs copy-on-write.
        """
        if not self.alloc.enable_prefix_caching:
            return [], []
        limit = (len(tokens) - 1) // self.block_size
        hits, hs = [], []
        parent = b""
        bs = self.block_size
        for i in range(limit):
            parent = block_hash(parent, tokens[i * bs:(i + 1) * bs])
            b = self.alloc.lookup(parent)
            if b is None:
                break
            hits.append(b)
            hs.append(parent)
        return hits, hs

    def can_allocate(self, tokens) -> bool:
        hits, _ = self.plan_prefix(tokens)
        # reviving an evictable hit consumes one of the "free" slots too
        revived = sum(1 for b in hits if self.alloc.ref[b] == 0)
        need = self.blocks_needed(len(tokens)) - len(hits) + revived
        return need <= self.alloc.num_free

    def allocate(self, seq_id: int, tokens) -> int:
        """Create the block table for a new sequence. Returns the number of
        leading tokens whose K/V are already cached (prefill can skip them)."""
        if seq_id in self.tables:
            raise ValueError(f"seq {seq_id} already has blocks")
        if not self.can_allocate(tokens):
            raise OutOfBlocks("not enough blocks for prompt")
        hits, hs = self.plan_prefix(tokens)
        self.alloc.prefix_queries += (len(tokens) - 1) // self.block_size
        self.alloc.prefix_hits += len(hits)
        for b in hits:
            self.alloc.incref(b)
        blocks = hits + [self.alloc.allocate()
                         for _ in range(self.blocks_needed(len(tokens)) - len(hits))]
        self.tables[seq_id] = BlockTable(blocks, hs)
        return len(hits) * self.block_size

    # -- growth -----------------------------------------------------------
    def can_append(self, seq_id: int, num_tokens: int) -> bool:
        """Can the sequence grow to ``num_tokens`` total tokens?"""
        t = self.tables[seq_id]
        need = self.blocks_needed(num_tokens) - len(t.blocks)
        # +1 for a possible copy-on-write of a shared partial last block
        if t.blocks and self.alloc.ref[t.blocks[-1]] > 1:
            need += 1
        return need <= self.alloc.num_free

    def ensure_capacity(self, seq_id: int, num_tokens: int, write_from: int
                        ) -> list[tuple[int, int]]:
        """Grow the table to hold ``num_tokens`` and make the blocks that will
        be written (positions ``write_from ..``) private.

        Returns (src, dst) block copies the cache must perform BEFORE the
        forward pass: copy-on-write of shared blocks.
        """
        t = self.tables[seq_id]
        copies = []
        bs = self.block_size
        first_written = write_from // bs
        for i in range(first_written, len(t.blocks)):
            b = t.blocks[i]
            if self.alloc.ref[b] > 1:
                nb = self.alloc.allocate()
                self.alloc.decref(b)
                t.blocks[i] = nb
                copies.append((b, nb))
                # the copy is about to diverge: it no longer matches the hash
                del t.hashes[i:]
        while len(t.blocks) < self.blocks_needed(num_tokens):
            t.blocks.append(self.alloc.allocate())
        return copies

    def mark_computed(self, seq_id: int, tokens, num_computed: int) -> None:
        """Register every newly *full* block in the prefix cache.

        Generated tokens are cached too, so a follow-up turn of a chat (prompt =
        previous prompt + answer) hits the cache all the way through.
        """
        t = self.tables[seq_id]
        bs = self.block_size
        full = num_computed // bs
        parent = t.hashes[-1] if t.hashes else b""
        for i in range(len(t.hashes), full):
            parent = block_hash(parent, tokens[i * bs:(i + 1) * bs])
            t.hashes.append(parent)
            self.alloc.register(t.blocks[i], parent)

    # -- fork / free ------------------------------------------------------
    def fork(self, parent_id: int, child_id: int) -> None:
        """Child shares every block of the parent (ref_count++); writes later
        trigger copy-on-write in :meth:`ensure_capacity`."""
        p = self.tables[parent_id]
        for b in p.blocks:
            self.alloc.incref(b)
        self.tables[child_id] = BlockTable(list(p.blocks), list(p.hashes))

    def free(self, seq_id: int) -> None:
        t = self.tables.pop(seq_id)
        # Free in reverse so that, among evictable blocks, a prefix's *tail*
        # is evicted before its head (LRU pops oldest first). Evicting the head
        # first would orphan the tail - it could never be hit again.
        for b in reversed(t.blocks):
            self.alloc.decref(b)

    def block_table(self, seq_id: int) -> list[int]:
        return self.tables[seq_id].blocks


# --------------------------------------------------------------------------
# The physical pool + attention over it
# --------------------------------------------------------------------------


class PagedKVPool:
    """Physical KV storage: (layers, num_blocks, block_size, kv_heads, head_dim)."""

    def __init__(self, cfg: ModelConfig, num_blocks: int, block_size: int, dtype=np.float32):
        self.block_size = block_size
        shape = (cfg.n_layers, num_blocks, block_size, cfg.n_kv_heads, cfg.head_dim)
        self.k = np.zeros(shape, dtype)
        self.v = np.zeros(shape, dtype)
        # Flat views (layer, slot, kvh, d): a token's slot = block*B + offset.
        self.k_flat = self.k.reshape(cfg.n_layers, num_blocks * block_size, *shape[3:])
        self.v_flat = self.v.reshape(cfg.n_layers, num_blocks * block_size, *shape[3:])

    def copy_block(self, src: int, dst: int) -> None:
        self.k[:, dst] = self.k[:, src]
        self.v[:, dst] = self.v[:, src]

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes


def slot_mapping(block_table: list[int], positions: np.ndarray, block_size: int) -> np.ndarray:
    """Physical slot for each logical position (vLLM's ``slot_mapping``)."""
    bt = np.asarray(block_table, dtype=np.int64)
    return bt[positions // block_size] * block_size + positions % block_size


def paged_attention(q: np.ndarray, k_pool: np.ndarray, v_pool: np.ndarray,
                    block_table: list[int], context_len: int, q_start: int) -> np.ndarray:
    """Attention that reads K/V block by block through the block table, with an
    online (streaming) softmax - the structure of the PagedAttention kernel.

    q: (n, H, d) for positions q_start..q_start+n-1
    k_pool, v_pool: (num_blocks, block_size, KVH, d) for one layer
    The full (n x context_len) score matrix is never materialised: we keep a
    running max ``m``, normaliser ``l`` and un-normalised output ``acc`` and
    rescale them when a later block raises the max. Memory is O(block_size).
    """
    n, H, d = q.shape
    _, B, KVH, _ = k_pool.shape
    g = H // KVH
    qg = q.reshape(n, KVH, g, d).transpose(1, 2, 0, 3) * (1.0 / np.sqrt(d))  # (KVH,g,n,d)
    m = np.full((KVH, g, n, 1), -np.inf, dtype=q.dtype)
    l = np.zeros((KVH, g, n, 1), dtype=q.dtype)
    acc = np.zeros((KVH, g, n, d), dtype=q.dtype)
    qpos = q_start + np.arange(n)[:, None]
    for bi in range(-(-context_len // B)):
        blk = block_table[bi]
        valid = min(B, context_len - bi * B)
        kb = k_pool[blk, :valid].transpose(1, 2, 0)          # (KVH, d, valid)
        vb = v_pool[blk, :valid].transpose(1, 0, 2)          # (KVH, valid, d)
        s = qg @ kb[:, None]                                  # (KVH, g, n, valid)
        kpos = bi * B + np.arange(valid)[None, :]
        s = np.where(kpos <= qpos, s, -np.inf)
        m_new = np.maximum(m, s.max(-1, keepdims=True))
        # A query row can be fully masked in this block (and in all previous
        # ones) - guard exp(-inf - -inf) = nan.
        safe = np.where(np.isfinite(m_new), m_new, 0.0)
        p = np.exp(s - safe)
        corr = np.exp(np.where(np.isfinite(m), m - safe, -np.inf))
        l = l * corr + p.sum(-1, keepdims=True)
        acc = acc * corr + p @ vb[:, None]
        m = m_new
    out = acc / l
    return out.transpose(2, 0, 1, 3).reshape(n, H, d)


class PagedBatch:
    """KVBackend for a batch of sequences whose K/V live in a :class:`PagedKVPool`.

    ``attn`` selects how attention reads the pool:

    * ``"gather"`` - gather the sequence's blocks into a contiguous view, then
      run dense attention. Fastest in numpy (one fancy-index, one matmul).
    * ``"paged"``  - :func:`paged_attention`, block-by-block online softmax.
      Slower here (a Python loop per block) but the shape of the real GPU kernel.
    Both are tested to match contiguous attention.
    """

    def __init__(self, pool: PagedKVPool, block_tables: list[list[int]],
                 starts: list[int], n_new: list[int], attn: str = "gather"):
        if attn not in ("gather", "paged"):
            raise ValueError(attn)
        self.pool, self.attn = pool, attn
        self.block_tables, self.starts, self.n_new = block_tables, starts, n_new
        self.offsets = np.concatenate([[0], np.cumsum(n_new)]).astype(int)
        B = pool.block_size
        self.positions = np.concatenate(
            [np.arange(s, s + n) for s, n in zip(starts, n_new)]).astype(np.int64)
        self.slots = np.concatenate([
            slot_mapping(bt, np.arange(s, s + n), B)
            for bt, s, n in zip(block_tables, starts, n_new)])
        # Precompute each sequence's gather index once per step, not per layer.
        self._ctx_slots = [slot_mapping(bt, np.arange(s + n), B)
                           for bt, s, n in zip(block_tables, starts, n_new)]

    def write_and_attend(self, layer, q, k, v):
        P = self.pool
        P.k_flat[layer, self.slots] = k
        P.v_flat[layer, self.slots] = v
        out = np.empty_like(q)
        for i, (a, b) in enumerate(zip(self.offsets[:-1], self.offsets[1:])):
            s = self.starts[i]
            if self.attn == "gather":
                cs = self._ctx_slots[i]
                out[a:b] = causal_attention(q[a:b], P.k_flat[layer, cs], P.v_flat[layer, cs], s)
            else:
                out[a:b] = paged_attention(q[a:b], P.k[layer], P.v[layer],
                                           self.block_tables[i], s + (b - a), s)
        return out
