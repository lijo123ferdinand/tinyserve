import numpy as np
import pytest

from tinyserve.kv_cache import KVCache, forward_with_caches
from tinyserve.model import causal_attention
from tinyserve.paged import (BlockAllocator, BlockManager, OutOfBlocks, PagedBatch,
                             PagedKVPool, block_hash, paged_attention)


@pytest.mark.parametrize("ctx,n,bs", [(1, 1, 4), (37, 1, 8), (37, 37, 8), (64, 5, 16), (9, 3, 4)])
def test_paged_attention_matches_contiguous(rng, ctx, n, bs):
    H, KVH, d = 4, 2, 8
    nblk = -(-ctx // bs)
    table = rng.permutation(nblk + 5)[:nblk].tolist()   # scattered physical blocks
    kp = rng.standard_normal((nblk + 5, bs, KVH, d))
    vp = rng.standard_normal((nblk + 5, bs, KVH, d))
    k = np.concatenate([kp[b] for b in table])[:ctx]
    v = np.concatenate([vp[b] for b in table])[:ctx]
    q = rng.standard_normal((n, H, d))
    ref = causal_attention(q, k, v, ctx - n)
    got = paged_attention(q, kp, vp, table, ctx, ctx - n)
    np.testing.assert_allclose(got, ref, atol=1e-12)


@pytest.mark.parametrize("attn", ["gather", "paged"])
def test_paged_model_matches_contiguous_cache(model, rng, attn):
    """Full model: prefill + several decode steps through a scattered block pool
    must reproduce the contiguous-cache logits."""
    bs = 4
    pool = PagedKVPool(model.cfg, num_blocks=64, block_size=bs, dtype=model.dtype)
    seqs = [rng.integers(0, model.cfg.vocab_size, n).tolist() for n in (6, 13, 1)]
    free = rng.permutation(64).tolist()
    tables = [[free.pop() for _ in range(10)] for _ in seqs]
    caches = [KVCache(model.cfg) for _ in seqs]
    starts = [0] * len(seqs)
    feed = seqs
    for _ in range(4):
        n_new = [len(f) for f in feed]
        kv = PagedBatch(pool, tables, starts, n_new, attn)
        got = model.forward(np.concatenate(feed), kv.positions, kv)
        ref = np.concatenate(forward_with_caches(model, caches, feed))
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)
        starts = [s + n for s, n in zip(starts, n_new)]
        feed = [[int(rng.integers(model.cfg.vocab_size))] for _ in seqs]


# ------------------------------------------------------------------ allocator

def test_allocator_free_list_and_refcount():
    a = BlockAllocator(3, enable_prefix_caching=False)
    b0, b1, b2 = a.allocate(), a.allocate(), a.allocate()
    assert a.num_free == 0
    with pytest.raises(OutOfBlocks):
        a.allocate()
    a.incref(b1)
    a.decref(b1)
    assert a.num_free == 0          # still referenced once
    a.decref(b1)
    assert a.num_free == 1 and a.allocate() == b1
    with pytest.raises(RuntimeError):
        a.decref(b0); a.decref(b0)  # double free


def test_fork_copy_on_write_copies_only_partial_block():
    bm = BlockManager(16, block_size=4)
    toks = list(range(10))                       # 2 full blocks + 2 tokens
    bm.allocate(0, toks)
    bm.fork(0, 1)
    shared = bm.block_table(0)
    assert bm.block_table(1) == shared and all(bm.alloc.ref[b] == 2 for b in shared)
    # seq 1 writes position 10 (into the partial 3rd block) -> exactly one copy
    copies = bm.ensure_capacity(1, 11, write_from=10)
    assert copies == [(shared[2], bm.block_table(1)[2])]
    assert bm.block_table(1)[:2] == shared[:2]   # full blocks stay shared
    assert bm.alloc.ref[shared[2]] == 1          # parent now owns the original alone
    assert bm.ensure_capacity(0, 11, write_from=10) == []   # no copy needed for parent


def test_cow_data_is_copied(model):
    pool = PagedKVPool(model.cfg, 4, 4)
    pool.k[:, 1] = 7.0
    pool.copy_block(1, 3)
    assert np.all(pool.k[:, 3] == 7.0)


def test_prefix_cache_hit_and_reuse():
    bm = BlockManager(32, block_size=4)
    prompt = list(range(13))                    # 3 full blocks + 1
    assert bm.allocate(0, prompt) == 0
    bm.mark_computed(0, prompt, 13)
    # same prefix, different tail: reuse 3 blocks
    other = prompt[:12] + [99, 98]
    assert bm.allocate(1, other) == 12
    assert bm.block_table(1)[:3] == bm.block_table(0)[:3]
    # the last prompt token is always recomputed: a 12-token prompt reuses 2 blocks
    assert bm.allocate(2, prompt[:12]) == 8


def test_prefix_cache_survives_free_until_evicted():
    bm = BlockManager(4, block_size=4)
    p = list(range(9))
    bm.allocate(0, p)
    bm.mark_computed(0, p, 9)
    bm.free(0)
    assert bm.alloc.num_free == 4               # evictable blocks count as free
    assert bm.allocate(1, p) == 8               # revived from the evictable set
    bm.free(1)
    # a different prompt needing all 4 blocks evicts the cached ones
    bm.allocate(2, [50 + i for i in range(16)])
    assert bm.alloc.cached == {}
    bm.free(2)
    assert bm.allocate(3, p) == 0


def test_block_hash_is_position_aware():
    a = block_hash(b"", [1, 2, 3, 4])
    b = block_hash(block_hash(b"", [9, 9, 9, 9]), [1, 2, 3, 4])
    assert a != b and a == block_hash(b"", [1, 2, 3, 4])


def test_prefix_caching_disabled():
    bm = BlockManager(16, 4, enable_prefix_caching=False)
    p = list(range(9))
    bm.allocate(0, p)
    bm.mark_computed(0, p, 9)
    assert bm.allocate(1, p) == 0


def test_can_append_accounts_for_cow():
    bm = BlockManager(3, block_size=4)
    bm.allocate(0, [1, 2, 3, 4, 5, 6])        # 2 blocks
    bm.fork(0, 1)
    assert bm.can_append(1, 7)                # COW of partial block needs 1, have 1
    bm.ensure_capacity(1, 7, 6)
    assert not bm.can_append(0, 9)            # needs a new block, none left
