"""OpenAI-compatible HTTP server (stdlib only).

    POST /v1/completions   {"prompt": "...", "max_tokens": 64, "stream": true, ...}
    GET  /v1/models
    GET  /health

Threading model: ``ThreadingHTTPServer`` gives each connection a thread, but
the engine is single-threaded by design (the scheduler owns global state, and
batching only works if one loop sees every request). So handler threads never
touch the engine directly: they enqueue a submission and then block on a
per-request ``queue.Queue`` of deltas. One engine thread drains submissions,
calls ``engine.step()`` - which advances *all* in-flight requests at once -
and fans the deltas out. That is the same shape as vLLM's AsyncLLMEngine.

A client that disconnects mid-stream is detected on the next write and its
request is aborted, freeing its KV blocks for everyone else.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .engine import LLMEngine, StepOutput
from .sampling import SamplingParams

_DONE = object()


class EngineLoop:
    """Owns the engine; the only thread that ever calls into it."""

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._submit: queue.Queue = queue.Queue()
        self._streams: dict[str, queue.Queue] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._submit.put(None)
        self._thread.join(timeout=5)

    def submit(self, prompt: str, params: SamplingParams) -> tuple[str, queue.Queue]:
        rid = f"cmpl-{uuid.uuid4().hex[:24]}"
        q: queue.Queue = queue.Queue()
        self._submit.put(("add", rid, prompt, params, q))
        return rid, q

    def abort(self, rid: str) -> None:
        self._submit.put(("abort", rid))

    def _drain(self, block: bool) -> None:
        try:
            item = self._submit.get(block=block, timeout=0.5 if block else None)
        except queue.Empty:
            return
        while item is not None:
            if item[0] == "add":
                _, rid, prompt, params, q = item
                try:
                    self.engine.add_request(prompt, params, request_id=rid)
                    self._streams[rid] = q
                except ValueError as e:     # bad request: report, don't crash the loop
                    q.put(e)
            else:
                rid = item[1]
                self.engine.abort(rid)
                self._streams.pop(rid, None)
            try:
                item = self._submit.get_nowait()
            except queue.Empty:
                item = None

    def _run(self) -> None:
        while not self._stop.is_set():
            # Block only when idle; while requests are in flight, never wait -
            # every millisecond spent sleeping is a millisecond of no decoding.
            self._drain(block=not self.engine.has_unfinished())
            if not self.engine.has_unfinished():
                continue
            for out in self.engine.step():
                q = self._streams.get(out.request_id)
                if q is not None:
                    q.put(out)
            live = self._unfinished()
            for rid in [r for r in self._streams if r not in live]:
                self._streams.pop(rid).put(_DONE)
                self.engine.release(rid)

    def _unfinished(self) -> set[str]:
        s = self.engine.scheduler
        return {x.request_id for x in (*s.waiting, *s.running)}


def _params_from_body(body: dict) -> SamplingParams:
    stop = body.get("stop") or ()
    if isinstance(stop, str):
        stop = (stop,)
    return SamplingParams(
        max_tokens=int(body.get("max_tokens", 16)),   # OpenAI's default
        temperature=float(body.get("temperature", 1.0)),
        top_k=int(body.get("top_k", 0)),
        top_p=float(body.get("top_p", 1.0)),
        repetition_penalty=float(body.get("repetition_penalty", 1.0)),
        seed=body.get("seed"),
        n=int(body.get("n", 1)),
        stop=tuple(stop),
    )


def make_handler(loop: EngineLoop, model_name: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep test output clean
            pass

        def _json(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, msg: str) -> None:
            self._json(code, {"error": {"message": msg, "type": "invalid_request_error"}})

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": model_name, "object": "model", "owned_by": "tinyserve"}]})
            else:
                self._error(404, f"no route {self.path}")

        def do_POST(self):
            if self.path != "/v1/completions":
                return self._error(404, f"no route {self.path}")
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                prompt = body["prompt"]
                if isinstance(prompt, list):
                    if len(prompt) != 1 or not isinstance(prompt[0], str):
                        raise ValueError("only a single string prompt is supported")
                    prompt = prompt[0]
                if not isinstance(prompt, str):
                    raise ValueError("prompt must be a string")
                params = _params_from_body(body)
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
                return self._error(400, f"bad request: {e}")
            rid, q = loop.submit(prompt, params)
            created = int(time.time())
            if body.get("stream"):
                self._stream(rid, q, created)
            else:
                self._blocking(rid, q, created, params.n, len(prompt))

        def _chunk(self, rid, created, choices) -> dict:
            return {"id": rid, "object": "text_completion", "created": created,
                    "model": model_name, "choices": choices}

        def _blocking(self, rid, q, created, n, prompt_chars):
            texts = [""] * n
            reasons = [None] * n
            completion_tokens = 0
            while (item := q.get()) is not _DONE:
                if isinstance(item, Exception):
                    return self._error(400, str(item))
                texts[item.index] += item.new_text
                completion_tokens += len(item.new_token_ids)
                if item.finished:
                    reasons[item.index] = item.finish_reason
            out = self._chunk(rid, created, [
                {"index": i, "text": texts[i], "logprobs": None, "finish_reason": reasons[i]}
                for i in range(n)])
            # char-level tokenizer: prompt tokens == prompt characters
            out["usage"] = {"prompt_tokens": prompt_chars, "completion_tokens": completion_tokens,
                            "total_tokens": prompt_chars + completion_tokens}
            self._json(200, out)

        def _stream(self, rid, q, created):
            first = q.get()
            if isinstance(first, Exception):
                return self._error(400, str(first))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            # No Content-Length on a stream: close the connection to delimit it.
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            item = first
            try:
                while item is not _DONE:
                    out: StepOutput = item
                    if out.new_text or out.finished:
                        ev = self._chunk(rid, created, [{
                            "index": out.index, "text": out.new_text, "logprobs": None,
                            "finish_reason": out.finish_reason}])
                        self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
                        self.wfile.flush()
                    item = q.get()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                loop.abort(rid)   # client went away: give its KV blocks back

    return Handler


class _HTTPServer(ThreadingHTTPServer):
    # socketserver's default listen backlog is 5: a burst of 8 simultaneous
    # clients gets connections reset by the OS before a handler ever runs.
    request_queue_size = 128
    daemon_threads = True


class Server:
    def __init__(self, engine: LLMEngine, host: str = "127.0.0.1", port: int = 8000,
                 model_name: str = "tinyserve-shakespeare"):
        self.loop = EngineLoop(engine)
        self.httpd = _HTTPServer((host, port), make_handler(self.loop, model_name))

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def start_background(self) -> "Server":
        self.loop.start()
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def serve_forever(self) -> None:
        self.loop.start()
        try:
            self.httpd.serve_forever()
        finally:
            self.loop.stop()

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.loop.stop()
