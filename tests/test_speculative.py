import numpy as np
import pytest

from tinyserve.sampling import SamplingParams, draw
from tinyserve.speculative import SpeculativeDecoder, autoregressive, verify

from conftest import tiny_model


def _dist(rng, V, sharp=1.0):
    x = np.exp(rng.standard_normal(V) * sharp)
    return x / x.sum()


@pytest.mark.parametrize("k", [1, 2, 3])
def test_rejection_sampling_reproduces_target_distribution(k):
    """Draft q and target p differ a lot; the emitted tokens must still be
    distributed exactly as p (here: i.i.d. over positions since p does not
    depend on context). Checks the first two emitted tokens' joint law."""
    rng = np.random.default_rng(k)
    V = 5
    p = np.stack([_dist(rng, V, 1.5) for _ in range(k + 1)])
    q = np.stack([_dist(rng, V, 1.5) for _ in range(k)])
    trials = 40000
    first = np.zeros(V)
    joint = np.zeros((V, V))
    n_joint = 0
    for _ in range(trials):
        xs = [draw(q[i], rng) for i in range(k)]
        out, _ = verify(p, q, xs, rng)
        first[out[0]] += 1
        if len(out) >= 2:
            joint[out[0], out[1]] += 1
            n_joint += 1
    tv = 0.5 * np.abs(first / trials - p[0]).sum()
    assert tv < 0.015, tv
    if k >= 2:
        # Given 2+ tokens were emitted, token 1 must still follow p[1] (indep. of token 0)
        cond = joint.sum(0) / n_joint
        assert 0.5 * np.abs(cond - p[1]).sum() < 0.03


def test_naive_greedy_acceptance_would_be_biased():
    """Sanity check of the test itself: a wrong rule (accept iff the draft token
    is the target's argmax, else take the argmax) is NOT the target distribution."""
    rng = np.random.default_rng(0)
    p, q = np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.3, 0.5])
    wrong = np.bincount([int(np.argmax(p)) for _ in range(1000)], minlength=3) / 1000
    assert 0.5 * np.abs(wrong - p).sum() > 0.4
    good = np.bincount([verify(p[None].repeat(2, 0), q[None], [draw(q, rng)], rng)[0][0]
                        for _ in range(20000)], minlength=3) / 20000
    assert 0.5 * np.abs(good - p).sum() < 0.02


def test_identical_models_accept_everything():
    rng = np.random.default_rng(1)
    p = np.stack([_dist(rng, 6) for _ in range(4)])
    for _ in range(200):
        xs = [draw(p[i], rng) for i in range(3)]
        out, acc = verify(p, p[:3], xs, rng)
        assert acc == 3 and out[:3] == xs and len(out) == 4


@pytest.mark.parametrize("k", [1, 3, 5])
def test_speculative_greedy_equals_target_greedy(model, tok, k):
    draft = tiny_model(tok.vocab_size, seed=99, layers=1)
    prompt = tok.encode("ROMEO:\nBut soft, what light")
    params = SamplingParams(max_tokens=40, temperature=0)
    want, _ = autoregressive(model, prompt, params)
    got, st = SpeculativeDecoder(model, draft, k).generate(prompt, params)
    assert got == want
    assert st.emitted == 40 and st.target_forwards <= 40


def test_speculative_self_draft_accepts_all(model, tok):
    prompt = tok.encode("To be")
    got, st = SpeculativeDecoder(model, model, 4).generate(
        prompt, SamplingParams(max_tokens=30, temperature=0.8, seed=1))
    assert st.acceptance_rate == 1.0 and len(got) == 30
    assert st.tokens_per_target_forward > 4


def test_speculative_respects_max_tokens_and_context(model, tok):
    draft = tiny_model(tok.vocab_size, seed=3, layers=1)
    for mt in (1, 2, 7):
        got, _ = SpeculativeDecoder(model, draft, 4).generate(
            [1, 2, 3], SamplingParams(max_tokens=mt, temperature=1.0, seed=mt))
        assert len(got) == mt
    long_prompt = [5] * (model.cfg.max_seq_len - 3)
    got, _ = SpeculativeDecoder(model, draft, 4).generate(
        long_prompt, SamplingParams(max_tokens=50, temperature=0))
    assert len(got) == 3
