import numpy as np
import pytest

from tinyserve.sampling import (SamplingParams, apply_repetition_penalty, draw, make_rng,
                                probs, sample)

LOGITS = np.array([2.0, 1.0, 0.5, -1.0, 3.0])


def test_greedy():
    assert sample(LOGITS, SamplingParams(temperature=0), make_rng(0, 0)) == 4
    assert probs(LOGITS, SamplingParams(temperature=0))[4] == 1.0


def test_temperature_one_is_softmax():
    e = np.exp(LOGITS - LOGITS.max())
    np.testing.assert_allclose(probs(LOGITS, SamplingParams()), e / e.sum())


def test_top_k_support():
    p = probs(LOGITS, SamplingParams(top_k=2))
    assert set(np.nonzero(p)[0]) == {0, 4} and p.sum() == pytest.approx(1)


def test_top_p_keeps_smallest_nucleus():
    p_full = probs(LOGITS, SamplingParams())
    p = probs(LOGITS, SamplingParams(top_p=0.8))
    order = np.argsort(-p_full)
    # p_full sorted: [~.61, ~.22, ...]: .61 < .8 so the second token is needed too
    assert set(np.nonzero(p)[0]) == set(order[:2])
    assert set(np.nonzero(probs(LOGITS, SamplingParams(top_p=0.1)))[0]) == {4}


def test_repetition_penalty_always_lowers_seen_tokens():
    out = apply_repetition_penalty(LOGITS, [0, 3], 2.0)
    assert out[0] == 1.0 and out[3] == -2.0 and out[4] == 3.0
    # greedy with a strong penalty avoids the repeated argmax
    assert sample(LOGITS, SamplingParams(temperature=0, repetition_penalty=10), None, [4]) == 0


def test_seeded_sampling_is_reproducible():
    p = SamplingParams(temperature=1.0, seed=42)
    a = [sample(LOGITS, p, r) for r in [make_rng(42, 0)] for _ in range(20)]
    b = [sample(LOGITS, p, r) for r in [make_rng(42, 0)] for _ in range(20)]
    assert a == b
    assert make_rng(42, 0).random() != make_rng(42, 1).random()


def test_draw_matches_distribution():
    p = np.array([0.1, 0.0, 0.6, 0.3])
    rng = np.random.default_rng(0)
    counts = np.bincount([draw(p, rng) for _ in range(20000)], minlength=4) / 20000
    assert counts[1] == 0
    np.testing.assert_allclose(counts, p, atol=0.015)


@pytest.mark.parametrize("bad", [dict(max_tokens=0), dict(temperature=-1), dict(top_p=0),
                                 dict(top_k=-1), dict(repetition_penalty=0), dict(n=0)])
def test_param_validation(bad):
    with pytest.raises(ValueError):
        SamplingParams(**bad)
