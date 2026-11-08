"""Command-line entry point: ``tinyserve {serve,generate,bench,train}``."""

from __future__ import annotations

import argparse
import sys

from . import load_pretrained
from .engine import EngineConfig, LLMEngine
from .sampling import SamplingParams


def _params(a) -> SamplingParams:
    return SamplingParams(max_tokens=a.max_tokens, temperature=a.temperature, top_k=a.top_k,
                          top_p=a.top_p, repetition_penalty=a.repetition_penalty, seed=a.seed,
                          n=getattr(a, "n", 1))


def cmd_generate(a) -> None:
    model, tok = load_pretrained(a.model)
    params = _params(a)
    if a.speculative:
        from .speculative import SpeculativeDecoder
        draft, _ = load_pretrained("draft")
        ids, st = SpeculativeDecoder(model, draft, a.speculative).generate(tok.encode(a.prompt),
                                                                           params)
        print(a.prompt + tok.decode(ids))
        print(f"\n[speculative k={a.speculative}] acceptance {st.acceptance_rate:.2f}, "
              f"{st.tokens_per_target_forward:.2f} tokens/target forward, {st.seconds:.2f}s",
              file=sys.stderr)
        return
    eng = LLMEngine(model, tok, EngineConfig())
    rid = eng.add_request(a.prompt, params)
    sys.stdout.write(a.prompt)
    while eng.has_unfinished():
        for o in eng.step():
            if params.n == 1:
                sys.stdout.write(o.new_text)
                sys.stdout.flush()
    if params.n > 1:
        for c in eng.result(rid):
            print(f"\n--- sample {c.index} ---\n{a.prompt}{c.text}")
    print()


def cmd_serve(a) -> None:
    from .server import Server
    model, tok = load_pretrained(a.model)
    eng = LLMEngine(model, tok, EngineConfig(num_blocks=a.num_blocks, block_size=a.block_size,
                                             max_num_seqs=a.max_num_seqs))
    srv = Server(eng, a.host, a.port)
    print(f"tinyserve listening on http://{a.host}:{srv.port}  (POST /v1/completions)",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_bench(a) -> None:
    from .bench import main
    main(a.which, a.json)


def cmd_train(a) -> None:
    from .train import train
    train(a.preset, a.steps, a.batch, a.seq, a.lr, a.seed, a.out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="tinyserve")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def sampling_args(p):
        p.add_argument("--model", default="target", choices=["target", "draft"])
        p.add_argument("--max-tokens", type=int, default=200)
        p.add_argument("--temperature", type=float, default=0.8)
        p.add_argument("--top-k", type=int, default=0)
        p.add_argument("--top-p", type=float, default=1.0)
        p.add_argument("--repetition-penalty", type=float, default=1.0)
        p.add_argument("--seed", type=int, default=None)

    g = sub.add_parser("generate", help="complete a prompt locally")
    g.add_argument("prompt")
    sampling_args(g)
    g.add_argument("-n", type=int, default=1, help="parallel samples (copy-on-write fork)")
    g.add_argument("--speculative", type=int, default=0, metavar="K",
                   help="use speculative decoding with K draft tokens")
    g.set_defaults(fn=cmd_generate)

    s = sub.add_parser("serve", help="OpenAI-compatible HTTP server")
    s.add_argument("--model", default="target", choices=["target", "draft"])
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--num-blocks", type=int, default=512)
    s.add_argument("--block-size", type=int, default=16)
    s.add_argument("--max-num-seqs", type=int, default=32)
    s.set_defaults(fn=cmd_serve)

    b = sub.add_parser("bench", help="run benchmarks")
    b.add_argument("which", nargs="?", default="all",
                   choices=["all", "batching", "memory", "prefix", "spec"])
    b.add_argument("--json", default=None, help="also write results to this file")
    b.set_defaults(fn=cmd_bench)

    t = sub.add_parser("train", help="train a char-level model on the bundled text")
    t.add_argument("--preset", choices=["target", "draft"], default="target")
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--seq", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", required=True)
    t.set_defaults(fn=cmd_train)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
