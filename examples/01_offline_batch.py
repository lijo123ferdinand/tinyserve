"""Offline batch generation: many prompts, one engine, continuous batching.

    python examples/01_offline_batch.py
"""

import time

from tinyserve import EngineConfig, LLMEngine, SamplingParams, load_pretrained

model, tok = load_pretrained("target")
engine = LLMEngine(model, tok, EngineConfig(num_blocks=256, block_size=16, max_num_seqs=8))

prompts = ["ROMEO:\n", "JULIET:\n", "KING HENRY VI:\n", "First Citizen:\n",
           "MENENIUS:\n", "LADY CAPULET:\n"]
params = SamplingParams(max_tokens=120, temperature=0.8, top_p=0.95, seed=42)

t0 = time.perf_counter()
results = engine.generate(prompts, params)
dt = time.perf_counter() - t0

for prompt, (out,) in zip(prompts, results):
    print("-" * 60)
    print(prompt + out.text)
st = engine.stats
print("=" * 60)
print(f"{st.generated_tokens} tokens in {dt:.2f}s ({st.generated_tokens / dt:.0f} tok/s), "
      f"{st.steps} engine steps, avg {st.generated_tokens / st.steps:.1f} sequences per step")
