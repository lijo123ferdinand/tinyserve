# tinyserve

**An LLM inference server built from scratch in Python + numpy, showing how vLLM-style serving works: paged KV cache, continuous batching, prefix caching, speculative decoding, and OpenAI-compatible streaming.**

[![CI](https://github.com/lijo123ferdinand/tinyserve/actions/workflows/ci.yml/badge.svg)](https://github.com/lijo123ferdinand/tinyserve/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![deps](https://img.shields.io/badge/deps-numpy%20only-green)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

**Measured on a laptop CPU:** paging cut KV-cache waste from 61.7% to 7.1% and fit 3.25x more concurrent
sequences in the same memory; continuous batching finished a bursty trace in 3.5x fewer forward
passes than static batching; prefix caching skipped 85% of prompt compute (2.18x faster). Details in
[Benchmarks](#benchmarks).

## Why

Production LLM servers (vLLM, TGI, SGLang, TensorRT-LLM) get most of their throughput from
**systems** ideas, not model ideas: how the KV cache is laid out in memory, when requests join a
batch, which work can be skipped or shared, and how many tokens one forward pass can be made to
produce. Those ideas are buried under CUDA kernels, tensor parallelism and many thousands of lines
of framework code.

tinyserve keeps only the systems part. It is small enough to read in an afternoon, and every
mechanism is real and tested:

| mechanism | where | what the tests prove |
|---|---|---|
| Llama-style decoder (RMSNorm, RoPE, GQA, SwiGLU) | `model.py` | matches a textbook attention; RoPE is relative |
| KV cache | `kv_cache.py` | cached decode logits == no-cache logits (allclose) |
| Paged KV cache: blocks, block tables, free list, ref counts | `paged.py` | paged attention == contiguous attention, for scattered blocks |
| Copy-on-write fork (`n > 1` parallel sampling) | `paged.py`, `engine.py` | only the partial last block is copied; sample 0 == the `n=1` run |
| Prefix caching (chained SHA-256 block hashes, LRU eviction) | `paged.py` | hits skip prefill; outputs unchanged |
| Continuous batching + token budget + preemption (recompute) | `scheduler.py` | batched == single; preempted runs give identical (even sampled) outputs; no leaked blocks |
| Sampling: greedy, temperature, top-k, top-p, repetition penalty, seeded | `sampling.py` | supports and distributions are right |
| Speculative decoding with the rejection-sampling acceptance rule | `speculative.py` | empirical output law == target law; greedy spec == greedy target |
| OpenAI `/v1/completions` with SSE streaming (stdlib `http.server`) | `server.py` | stream == blocking result; concurrent clients batched |
| The training code for the bundled models (hand-written backward) | `train.py` | gradients == finite differences |

The bundled models are a 0.76M-parameter, 4-layer char-level transformer (the *target*) and a
0.055M-parameter, 1-layer one (the *draft*), both trained in numpy on Tiny Shakespeare by
`train.py`. They write pseudo-Shakespeare, which makes a broken cache visible to the naked eye
and makes speculative decoding meaningful: two random models never agree; two trained ones often do.

## Quickstart

```bash
git clone https://github.com/lijo123ferdinand/tinyserve && cd tinyserve
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

```
$ pytest -q
........................................................................ [ 93%]
.....                                                                    [100%]
77 passed in 2.66s
```

Generate locally (streams token by token through the continuous-batching engine):

```
$ tinyserve generate "ROMEO:" --max-tokens 200 --temperature 0.8 --seed 1
ROMEO:
Were you there be for your crown, we have done.

QUEEN ELIZABETH:
Sone which to ere as he did man's love,
At happy here to me be here to the death.

QUEEN ELIZABETH:
Why, sir, it silvary this safes f
```

Serve the OpenAI-compatible API:

```
$ tinyserve serve --port 8765
tinyserve listening on http://127.0.0.1:8765  (POST /v1/completions)

$ curl -s http://127.0.0.1:8765/v1/completions -H 'Content-Type: application/json' \
    -d '{"prompt": "JULIET:\n", "max_tokens": 60, "temperature": 0.7, "seed": 3}'
{"id": "cmpl-5fdb4f77059c4d0c99a9b27a", "object": "text_completion", "created": 1794117891, "model": "tinyserve-shakespeare", "choices": [{"index": 0, "text": "All even good ross i' the libering person,\nTherefore him he ", "logprobs": null, "finish_reason": "length"}], "usage": {"prompt_tokens": 8, "completion_tokens": 60, "total_tokens": 68}}

$ curl -sN http://127.0.0.1:8765/v1/completions -H 'Content-Type: application/json' \
    -d '{"prompt": "JULIET:\n", "max_tokens": 4, "stream": true, "seed": 3}'
data: {"id": "cmpl-5ff684d07cfd4371a19c8c52", "object": "text_completion", "created": 1794117891, "model": "tinyserve-shakespeare", "choices": [{"index": 0, "text": "A", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-5ff684d07cfd4371a19c8c52", "object": "text_completion", "created": 1794117891, "model": "tinyserve-shakespeare", "choices": [{"index": 0, "text": "h", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-5ff684d07cfd4371a19c8c52", "object": "text_completion", "created": 1794117891, "model": "tinyserve-shakespeare", "choices": [{"index": 0, "text": ",", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-5ff684d07cfd4371a19c8c52", "object": "text_completion", "created": 1794117891, "model": "tinyserve-shakespeare", "choices": [{"index": 0, "text": " ", "logprobs": null, "finish_reason": "length"}]}

data: [DONE]
```

Speculative decoding (`examples/03_speculative.py`). Greedy output, which repeats as greedy decoding of small models does:

```
$ python examples/03_speculative.py
KING RICHARD III:
Now is the winter of our discontent
That the senators of the senators of the state
That the senators of the senators of the state
...
identical to plain greedy decoding: True
acceptance rate 0.31; 2.21 tokens per target forward (68 target forwards for 150 tokens)
wall time: autoregressive 0.055s, speculative 0.049s -> 1.12x
```

Other examples: `01_offline_batch.py` (six prompts batched: `720 tokens in 0.13s (5673 tok/s), 120 engine
steps, avg 6.0 sequences per step`), `02_stream_client.py` (an SSE client), and
`04_paged_kv_walkthrough.py`, which prints block tables while prefix sharing and copy-on-write happen.

## Architecture

```
  HTTP clients                       tinyserve process
  ────────────            ┌──────────────────────────────────────────────────────────┐
  POST /v1/completions ──►│ handler thread ─┐                                        │
  POST ... stream=true ──►│ handler thread ─┤ submit queue                           │
                          │      ▲          ▼                                        │
                          │      │   ┌────────────── engine thread ───────────────┐  │
                          │ per-request  │ loop:                                  │  │
                          │ delta queue  │   Scheduler.schedule()                 │  │
                          │      │       │     running decodes (oldest first)     │  │
                          │      │       │     ├─ no room? preempt youngest       │  │
                          │      │       │     └─ admit waiting under token budget│  │
                          │      │       │   apply copy-on-write block copies     │  │
                          │      │       │   one flat batch: prefills + decodes   │  │
                          │      │       │   Transformer.forward ──► PagedBatch   │  │
                          │      │       │       (slot_mapping write, gather /    │  │
                          │      │       │        paged attention read)           │  │
                          │      └───────│   sample per sequence (own seeded RNG) │  │
                          │              │   publish full blocks to prefix cache  │  │
                          │              └────────────────────────────────────────┘  │
                          │                                                          │
                          │  BlockManager: seq -> block table   BlockAllocator:      │
                          │   seq A [ 7][ 2][ 9]                 free list (deque)   │
                          │   seq B [ 7][ 4]   <- block 7 shared ref counts          │
                          │                                      hash -> block (LRU) │
                          │  PagedKVPool: K,V [layers, blocks, block_size, kvh, d]   │
                          └──────────────────────────────────────────────────────────┘
```

One engine step = one forward pass over a *flat* batch: rows from every scheduled sequence are
concatenated, so a 60-token prefill and fifteen 1-token decodes share the same matmuls. The model
never sees the cache layout: it hands each layer's q/k/v to a `KVBackend`
(`NoCache`, `ContiguousBatch` or `PagedBatch`), the same seam vLLM draws between the model and
its cache engine.

A sequence's uncomputed tokens are always `tokens[num_computed:]`. A new request has its whole
prompt uncomputed (minus prefix-cache hits), a running one has exactly one token, and a preempted
one has prompt + output again. Prefill, decode and recompute are one code path.

## Design decisions and trade-offs

- **Recompute preemption, not swap.** When the pool runs dry, the youngest running sequence
  loses all its blocks and goes back to the front of the queue. There is no CPU tier to swap to,
  and re-prefilling a few hundred tokens costs one batched forward. Because every sequence carries
  its own RNG, a preempted sequence produces exactly the tokens it would have produced anyway.
- **No admissions in a step that preempted.** Admitting new work into a pool that just ran dry
  only causes the next preemption (thrashing).
- **FCFS admission that does not skip ahead.** If the head of the queue does not fit, nothing
  behind it is admitted. Skipping ahead raises throughput a little and starves long prompts.
- **The last prompt token is always recomputed.** Its logits are needed to sample the first
  output token. As a side effect, prefix-cache hits are only ever *full, read-only* blocks, so
  prefix sharing never needs copy-on-write; only `n > 1` forks do.
- **Chained SHA-256 block hashes.** A block's key covers every token before it, because the same
  8 characters at position 0 and at position 200 have different K/V (RoPE, and different
  history). A cryptographic hash rules out the "silently wrong output" failure a 64-bit `hash()`
  collision would cause.
- **Freed blocks are evicted tail-first.** A sequence's blocks are released in reverse, so LRU
  eviction removes the end of a cached prefix before its head; evicting the head first would
  orphan the rest.
- **Two paged attention paths.** `gather` (index the pool through the block table, then dense
  attention) is the fast path in numpy. `paged` walks blocks with an online softmax, the structure
  of the real PagedAttention kernel, and is O(block_size) in memory. Both are tested against
  contiguous attention.
- **Stop strings are held back while streaming.** If the stop string is `"END"` and the text so
  far ends in `"EN"`, those two characters are not streamed yet, since nothing streamed can be
  taken back.
- **Speculative decoding uses the processed distribution.** Acceptance compares the target and
  draft distributions *after* repetition penalty, temperature, top-k and top-p, so the output is
  distributed exactly as non-speculative sampling with the same parameters, not just under
  greedy decoding.
- **max_seq_len is the trained context (256).** RoPE positions never seen in training are out of
  distribution; a first version trained on 128-token windows visibly fell apart after ~130
  characters, so the advertised context is now exactly what the models were trained on.
- **A generous static baseline.** The static-batching policy uses the same flat, unpadded batches
  as continuous batching; real static servers also pad to the longest sequence. The measured gap
  is therefore a lower bound.

## Benchmarks

Measured on an Apple M1 laptop (CPU, numpy 2.5, Python 3.14) with `tinyserve bench all`. Every number
below was printed by that command; rerun it to get your own. Wall-clock ratios move by a few tenths
between runs (continuous/static throughput was 1.52x in one run and 1.29x in the run below), so the
hardware-independent columns (engine steps, slots, tokens per forward) matter most.

### Continuous vs static batching

128 requests, Poisson arrivals at 400 req/s, prompts of 16 to 96 characters, output lengths
log-uniform from 8 to 160 tokens, at most 16 concurrent sequences, greedy. Time is simulated:
arrivals follow the seeded trace, and the clock advances by the measured wall time of each engine
step.

| policy | throughput (tok/s) | mean latency (s) | p99 latency (s) | mean TTFT (s) | engine steps | mean tokens/step |
|---|---|---|---|---|---|---|
| static | 4809 | 0.56 | 1.02 | 0.52 | 1960 | 3.4 |
| continuous | 6207 | 0.41 | 0.74 | 0.29 | 553 | 12.1 |

Continuous batching produced the same tokens in **3.5x fewer forward passes** (553 vs 1960 steps),
for 1.29x throughput and 1.37x lower mean latency in this run. Under static batching, a batch keeps
running at 1 to 3 live sequences while its longest member finishes, and new arrivals wait. The
throughput gain is smaller than the step reduction because on CPU a step's cost still grows with
batch size; on a GPU decoding is memory-bound and a bigger batch is close to free.

### KV cache memory: contiguous reservation vs paged

Same trace through the paged engine (pool of 256 blocks x 16 slots = 4096 token slots,
`max_model_len` 256). At every step, the slots held by paged allocation are compared with what
reserving `max_model_len` per running sequence would hold.

| scheme | slots holding real tokens | wasted (reserved, never used) | max concurrent seqs in a pool of 4096 slots |
|---|---|---|---|
| contiguous (reserve max_model_len) | 38.3% | 61.7% | 16 |
| paged | 92.9% | 7.1% | 52 (measured) |

Paging cut KV waste from **61.7% to 7.1%** and fit **3.25x more concurrent sequences** in the same
memory. The remaining 7% is the half-empty last block of each sequence. GQA (2 KV heads for 4
query heads) halves the cache again: 2048 bytes per token.

### Prefix caching

32 requests sharing a 192-character system prompt plus a short user turn, 16 output tokens each.

| | prompt tokens computed | served from cache | wall time (s) |
|---|---|---|---|
| no prefix cache | 6765 | 0 | 0.28 |
| prefix cache | 1005 | 5760 (85%) | 0.13 |

**2.18x faster**, with greedy outputs byte-identical with and without the cache.

### Speculative decoding

Target: 0.76M parameters, 4 layers (validation loss 1.53 nats/char). Draft: 0.055M parameters,
1 layer (1.87). 6 prompts x 160 tokens.

| sampling | k | acceptance rate | tokens / target forward | wall speedup |
|---|---|---|---|---|
| greedy | 1 | 0.69 | 1.68 | 1.00x |
| greedy | 2 | 0.46 | 1.92 | 0.94x |
| greedy | 4 | 0.32 | 2.25 | 0.88x |
| greedy | 6 | 0.25 | 2.44 | 0.74x |
| T=0.8 | 1 | 0.68 | 1.68 | 0.90x |
| T=0.8 | 2 | 0.64 | 2.26 | 1.07x |
| T=0.8 | 4 | 0.45 | 2.75 | 1.07x |
| T=0.8 | 6 | 0.37 | 3.19 | 0.98x |

Measured forward costs: target, 1 token 272 µs; target, 5 tokens 373 µs; draft, 1 token 70 µs.

Speculation does what it promises on the metric that carries over to hardware: **up to 3.19
tokens per target forward**, so up to 3.2x fewer target forwards. **It does not pay off in wall time
here, and the measured costs show why.** The speedup is roughly
`tokens_per_forward / (k * c_draft/c_target + c_verify/c_target)`. A 1-layer draft should cost
about 1/4 of the 4-layer target, and it does (0.26), but only because both are dominated by
per-call Python overhead, not FLOPs. With k=4 the draft alone costs about one target forward, and
verifying 5 tokens costs 1.37x a single-token forward. Real deployments get their 2-3x because the
draft is 10-100x cheaper than a large, memory-bound target. This repo shows the mechanism and its
exactness (the tests check that the output distribution equals the target's), and the numbers above
show the cost condition it depends on.

## Limitations

- **CPU and numpy, so absolute speed is tiny.** A step is dominated by Python and per-layer
  overhead, not FLOPs. The *ratios* (batching, speculation) come from the same effects as on a
  GPU, where batching amortises weight reads and Python overhead plays the part of kernel launches,
  but the magnitudes differ.
- One engine thread and no tensor or pipeline parallelism; prefill and decode share the GIL.
- No chunked prefill: a prompt is prefilled in one step, so `max_num_batched_tokens` must be at
  least `max_model_len`.
- Speculative decoding runs on a single sequence with contiguous caches; it is not wired into the
  continuous-batching engine.
- Char-level tokenizer and a 0.76M-parameter model: the text is Shakespeare-flavoured, not
  coherent. `logprobs` and `echo` are not implemented.
- Preemption only recomputes; there is no swap-to-CPU path.

## Roadmap

- [ ] Chunked prefill (mix prompt chunks into decode steps to cap per-step latency).
- [ ] Speculative decoding inside the batched engine (per-sequence k, paged rollback).
- [ ] Swap-based preemption with a CPU block pool, and a cost model that picks swap vs recompute.
- [ ] `logprobs`, `echo`, and `/v1/chat/completions`.
- [ ] Priority / fair-share scheduling, and SLO-aware admission.
- [ ] A BPE tokenizer with incremental UTF-8-safe detokenization.
- [ ] Medusa / lookahead-style draft-free speculation.

## Layout

```
src/tinyserve/
  model.py        Llama-style transformer, KVBackend protocol, reference attention
  kv_cache.py     contiguous per-sequence KV cache (+ truncate for speculative rollback)
  paged.py        BlockAllocator, BlockManager (tables, COW, prefix cache), PagedKVPool, paged attention
  scheduler.py    continuous / static scheduling, token budget, preemption
  engine.py       LLMEngine: the step loop, sampling, forking, stop strings, streaming deltas
  sampling.py     SamplingParams and the logits processing pipeline
  speculative.py  draft-then-verify with rejection sampling
  server.py       OpenAI-compatible HTTP + SSE server
  train.py        numpy training with a hand-derived backward pass
  bench.py        the benchmarks below
  weights/        bundled target.npz and draft.npz
examples/         offline batch, streaming client, speculative decoding, paged KV walkthrough
tests/            pytest suite
```

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023.
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022.
- Leviathan, Kalman, Matias, *Fast Inference from Transformers via Speculative Decoding*, ICML 2023.
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023.
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023.

## License

MIT, copyright 2026 Lijo Ferdinand.
