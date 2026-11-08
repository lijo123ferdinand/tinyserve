import pytest

from tinyserve import WEIGHTS_DIR, load_pretrained
from tinyserve.bench import make_trace, run_trace
from tinyserve.engine import EngineConfig, LLMEngine
from tinyserve.sampling import SamplingParams

needs_weights = pytest.mark.skipif(not (WEIGHTS_DIR / "target.npz").exists(),
                                   reason="bundled weights not built")


@needs_weights
def test_pretrained_model_writes_english_ish_text():
    model, tok = load_pretrained("target")
    out = LLMEngine(model, tok).generate(["ROMEO:\n"], SamplingParams(max_tokens=200,
                                                                     temperature=0))[0][0]
    words = out.text.split()
    common = {"the", "and", "to", "of", "I", "my", "a", "you", "that", "in", "with", "is", "not"}
    assert sum(w.strip(",.;:!?").lower() in common or w in common for w in words) >= 5


@needs_weights
def test_pretrained_draft_loads():
    d, tok = load_pretrained("draft")
    t, tok2 = load_pretrained("target")
    assert tok.chars == tok2.chars and d.cfg.vocab_size == t.cfg.vocab_size
    with pytest.raises(FileNotFoundError):
        load_pretrained("nope")


def test_trace_runner_smoke(model, tok):
    trace = make_trace(n=12, rate=200.0, seed=1, out_len=(4, 32))
    for policy in ("static", "continuous"):
        eng = LLMEngine(model, tok, EngineConfig(num_blocks=256, max_num_seqs=4, policy=policy))
        r = run_trace(eng, trace, sample_memory=True)
        assert r["requests"] == 12
        assert r["generated_tokens"] == sum(t.max_tokens for t in trace)
        assert r["throughput_tok_s"] > 0 and r["latency_p99_s"] >= r["latency_p50_s"]
        assert all(stored <= slots for _, stored, slots in r["memory_samples"])
