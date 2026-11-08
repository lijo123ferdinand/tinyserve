import numpy as np
import pytest

from tinyserve.engine import EngineConfig, LLMEngine
from tinyserve.sampling import SamplingParams

PROMPTS = ["First Citizen:\nBefore we proceed", "ROMEO:\nBut soft", "To be, or not",
           "KING RICHARD III:\nNow is the winter of our discontent", "A", "Speak, speak."]


def engine(model, tok, **kw):
    cfg = dict(num_blocks=128, block_size=8, max_num_seqs=8, max_num_batched_tokens=256)
    cfg.update(kw)
    return LLMEngine(model, tok, EngineConfig(**cfg))


def texts(outs):
    return [o[0].text for o in outs]


def reference_greedy(model, tok, prompt, n):
    """Uncached greedy decoding: the oracle every engine configuration must match."""
    ids = tok.encode(prompt)
    for _ in range(n):
        ids.append(int(np.argmax(model.logits_full(ids)[-1])))
    return tok.decode(ids[len(tok.encode(prompt)):])


def test_engine_greedy_matches_uncached_reference(model, tok):
    p = SamplingParams(max_tokens=24, temperature=0)
    got = texts(engine(model, tok).generate(PROMPTS[:3], p))
    assert got == [reference_greedy(model, tok, x, 24) for x in PROMPTS[:3]]


def test_batched_matches_single(model, tok):
    p = SamplingParams(max_tokens=30, temperature=0.9, top_k=20, seed=3)
    batched = texts(engine(model, tok).generate(PROMPTS, p))
    single = [texts(engine(model, tok).generate([x], p))[0] for x in PROMPTS]
    assert batched == single


@pytest.mark.parametrize("attn", ["gather", "paged"])
def test_attention_backends_agree(model, tok, attn):
    p = SamplingParams(max_tokens=20, temperature=0)
    assert texts(engine(model, tok, attn=attn).generate(PROMPTS, p)) == \
        texts(engine(model, tok).generate(PROMPTS, p))


def test_continuous_admits_mid_flight_and_static_does_not(model, tok):
    for policy, expect_overlap in (("continuous", True), ("static", False)):
        eng = engine(model, tok, policy=policy)
        a = eng.add_request("long one", SamplingParams(max_tokens=20, temperature=0))
        eng.step(); eng.step()
        b = eng.add_request("late", SamplingParams(max_tokens=3, temperature=0))
        eng.step()
        running = {s.request_id for s in eng.scheduler.running}
        assert (b in running) == expect_overlap and a in running


def test_static_and_continuous_same_outputs(model, tok):
    p = SamplingParams(max_tokens=25, temperature=0)
    assert texts(engine(model, tok, policy="static", max_num_seqs=2).generate(PROMPTS, p)) == \
        texts(engine(model, tok).generate(PROMPTS, p))


def test_preemption_recomputes_and_preserves_outputs(model, tok):
    """A pool too small for everyone forces preemption; outputs (even sampled
    ones, since each sequence carries its own RNG) must be unchanged."""
    p = SamplingParams(max_tokens=60, temperature=0.8, seed=11)
    roomy = engine(model, tok)
    want = texts(roomy.generate(PROMPTS, p))
    tight = engine(model, tok, num_blocks=40, block_size=4, max_model_len=128,
                   max_num_batched_tokens=128)
    outs = tight.generate(PROMPTS, p)
    assert tight.scheduler.num_preemptions > 0
    assert sum(o[0].num_preemptions for o in outs) == tight.scheduler.num_preemptions
    assert texts(outs) == want
    assert tight.bm.alloc.num_used == 0           # no leaked blocks


def test_prefix_caching_saves_prefill_and_preserves_outputs(model, tok):
    system = "You are a helpful Shakespearean assistant. Answer in verse.\n" * 2
    prompts = [system + q for q in ("Who art thou?", "What news?", "Speak!")]
    p = SamplingParams(max_tokens=12, temperature=0)
    off = engine(model, tok, enable_prefix_caching=False)
    on = engine(model, tok, enable_prefix_caching=True, max_num_seqs=1)  # serial => hits
    assert texts(on.generate(prompts, p)) == texts(off.generate(prompts, p))
    assert on.stats.cached_prompt_tokens >= 2 * (len(system) // 8) * 8
    assert on.stats.prefill_tokens < off.stats.prefill_tokens
    assert on.bm.alloc.prefix_hits > 0


def test_parallel_sampling_forks_with_copy_on_write(model, tok):
    eng = engine(model, tok)
    rid = eng.add_request("ROMEO:\nBut soft", SamplingParams(n=3, max_tokens=20,
                                                            temperature=1.0, seed=5))
    while eng.has_unfinished():
        eng.step()
    outs = eng.result(rid)
    assert [o.index for o in outs] == [0, 1, 2]
    assert len({o.text for o in outs}) > 1                 # independent samples
    assert eng.stats.cow_copies >= 2                       # partial last block copied
    # sample 0 is exactly what an n=1 request with the same seed produces
    solo = texts(engine(model, tok).generate(["ROMEO:\nBut soft"], SamplingParams(
        max_tokens=20, temperature=1.0, seed=5)))[0]
    assert outs[0].text == solo
    assert eng.bm.alloc.num_used == 0


def test_stop_strings_and_streaming(model, tok):
    ref = reference_greedy(model, tok, "To be, or not", 40)
    stop = ref[10:13]
    eng = engine(model, tok)
    rid = eng.add_request("To be, or not", SamplingParams(max_tokens=40, temperature=0,
                                                         stop=(stop,)))
    streamed = ""
    while eng.has_unfinished():
        for o in eng.step():
            streamed += o.new_text
            if o.finished:
                assert o.finish_reason == "stop"
    final = eng.result(rid)[0].text
    assert final == ref[:ref.find(stop)] and streamed == final
    assert stop not in final


def test_max_tokens_and_length_finish(model, tok):
    out = engine(model, tok).generate(["abc"], SamplingParams(max_tokens=7, temperature=0))[0][0]
    assert len(out.token_ids) == 7 and out.finish_reason == "length"
    out = engine(model, tok, max_model_len=64).generate(
        ["x" * 60], SamplingParams(max_tokens=50, temperature=0))[0][0]
    assert len(out.token_ids) == 4 and out.finish_reason == "length"


def test_abort_frees_blocks(model, tok):
    eng = engine(model, tok)
    rid = eng.add_request("hello there", SamplingParams(max_tokens=100))
    eng.step(); eng.step()
    assert eng.bm.alloc.num_used > 0
    eng.abort(rid)
    assert eng.bm.alloc.num_used == 0 and not eng.has_unfinished()


def test_token_budget_limits_admission(model, tok):
    eng = engine(model, tok, max_num_batched_tokens=256, max_model_len=256)
    for _ in range(4):
        eng.add_request("y" * 100, SamplingParams(max_tokens=2))
    eng.step()
    assert len(eng.scheduler.running) == 2       # 2 x 100 fit in 256, the 3rd does not


def test_invalid_requests(model, tok):
    eng = engine(model, tok)
    with pytest.raises(ValueError):
        eng.add_request("", SamplingParams())
    with pytest.raises(ValueError):
        eng.add_request("z" * 1000, SamplingParams())
    with pytest.raises(ValueError):
        LLMEngine(model, tok, EngineConfig(num_blocks=2, block_size=4))
