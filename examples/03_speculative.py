"""Speculative decoding: a 1-layer draft proposes, the 4-layer target verifies.

    python examples/03_speculative.py
"""

from tinyserve import SamplingParams, SpeculativeDecoder, load_pretrained
from tinyserve.speculative import autoregressive

target, tok = load_pretrained("target")
draft, _ = load_pretrained("draft")
prompt = tok.encode("KING RICHARD III:\nNow is the winter of our discontent\n")

greedy = SamplingParams(max_tokens=150, temperature=0)
base, t_base = autoregressive(target, prompt, greedy)
spec, st = SpeculativeDecoder(target, draft, k=4).generate(prompt, greedy)

print(tok.decode(prompt) + tok.decode(spec))
print()
print(f"identical to plain greedy decoding: {spec == base}")
print(f"acceptance rate {st.acceptance_rate:.2f}; {st.tokens_per_target_forward:.2f} tokens "
      f"per target forward ({st.target_forwards} target forwards for {st.emitted} tokens)")
print(f"wall time: autoregressive {t_base:.3f}s, speculative {st.seconds:.3f}s "
      f"-> {t_base / st.seconds:.2f}x")
