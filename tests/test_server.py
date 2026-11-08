import json
import threading
import urllib.error
import urllib.request

import pytest

from tinyserve.engine import EngineConfig, LLMEngine
from tinyserve.server import Server


@pytest.fixture(scope="module")
def server(model, tok):
    eng = LLMEngine(model, tok, EngineConfig(num_blocks=256, block_size=8))
    srv = Server(eng, port=0).start_background()
    yield srv
    srv.shutdown()


def post(srv, body, raw=False):
    req = urllib.request.Request(f"http://127.0.0.1:{srv.port}/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=30)
    return resp if raw else json.loads(resp.read())


def sse_events(resp):
    events = []
    for line in resp.read().decode().split("\n\n"):
        if line.startswith("data: "):
            events.append(line[6:])
    return events


def test_health_and_models(server):
    base = f"http://127.0.0.1:{server.port}"
    assert json.loads(urllib.request.urlopen(base + "/health").read())["status"] == "ok"
    models = json.loads(urllib.request.urlopen(base + "/v1/models").read())
    assert models["data"][0]["id"] == "tinyserve-shakespeare"


def test_completion(server):
    r = post(server, {"prompt": "ROMEO:", "max_tokens": 12, "temperature": 0})
    assert r["object"] == "text_completion"
    ch = r["choices"][0]
    assert len(ch["text"]) == 12 and ch["finish_reason"] == "length"
    assert r["usage"] == {"prompt_tokens": 6, "completion_tokens": 12, "total_tokens": 18}


def test_streaming_matches_blocking(server):
    body = {"prompt": "To be, or not", "max_tokens": 20, "temperature": 0}
    full = post(server, body)["choices"][0]["text"]
    resp = post(server, {**body, "stream": True}, raw=True)
    assert resp.headers["Content-Type"] == "text/event-stream"
    events = sse_events(resp)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert len(chunks) > 1                                  # actually incremental
    assert "".join(c["choices"][0]["text"] for c in chunks) == full
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_concurrent_requests_are_batched(server):
    results = {}

    def go(i):
        results[i] = post(server, {"prompt": f"Speak {i}", "max_tokens": 30, "temperature": 0.7,
                                   "seed": i})["choices"][0]["text"]

    threads = [threading.Thread(target=go, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 8 and all(len(t) == 30 for t in results.values())
    # seeded => identical to a sequential re-run
    assert post(server, {"prompt": "Speak 3", "max_tokens": 30, "temperature": 0.7,
                         "seed": 3})["choices"][0]["text"] == results[3]


def test_n_and_stop(server):
    r = post(server, {"prompt": "A", "max_tokens": 10, "n": 2, "seed": 1})
    assert [c["index"] for c in r["choices"]] == [0, 1]
    base = post(server, {"prompt": "B", "max_tokens": 20, "temperature": 0})["choices"][0]["text"]
    stop = base[5:7]
    r = post(server, {"prompt": "B", "max_tokens": 20, "temperature": 0, "stop": stop})
    assert r["choices"][0]["text"] == base[:base.find(stop)]
    assert r["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("body", [{}, {"prompt": 5}, {"prompt": "x", "temperature": -1},
                                  {"prompt": ""}])
def test_bad_requests(server, body):
    with pytest.raises(urllib.error.HTTPError) as e:
        post(server, body)
    assert e.value.code == 400
