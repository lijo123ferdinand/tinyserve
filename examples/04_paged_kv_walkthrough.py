"""Watch the paged KV cache work: block tables, prefix sharing, copy-on-write.

    python examples/04_paged_kv_walkthrough.py
"""

from tinyserve import EngineConfig, LLMEngine, SamplingParams, load_pretrained

model, tok = load_pretrained("target")
eng = LLMEngine(model, tok, EngineConfig(num_blocks=64, block_size=8, max_num_seqs=8))
bm = eng.bm


def show(title):
    print(f"\n== {title}")
    for s in eng.scheduler.running:
        refs = [int(bm.alloc.ref[b]) for b in bm.block_table(s.seq_id)]
        print(f"  seq {s.seq_id} ({s.request_id}#{s.index}) len={len(s.tokens):3d} "
              f"blocks={bm.block_table(s.seq_id)} refcounts={refs}")
    print(f"  used blocks {bm.alloc.num_used}/{bm.alloc.num_blocks}, "
          f"cached prefixes {len(bm.alloc.cached)}, COW copies so far {eng.stats.cow_copies}")


system = "You speak only in the manner of the Bard.\n"   # 42 chars -> 5 full blocks of 8
eng.add_request(system + "ROMEO:", SamplingParams(max_tokens=40, seed=1))
eng.step()
show("after prefilling request A (its full blocks are now published to the prefix cache)")

eng.add_request(system + "JULIET:", SamplingParams(max_tokens=40, seed=2))
eng.step()
show("request B shares A's system-prompt blocks (refcount 2) - only its tail was prefilled")

eng.add_request("To be, or not", SamplingParams(n=3, max_tokens=40, temperature=1.0, seed=3))
eng.step()
show("request C with n=3: three sequences share one prompt (refcount 3)")
eng.step()
show("one decode step later: the shared partial block was copied on write")

while eng.has_unfinished():
    eng.step()
print(f"\nall done: used blocks {bm.alloc.num_used}; prompt tokens served from cache: "
      f"{eng.stats.cached_prompt_tokens}")
