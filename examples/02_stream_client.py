"""Start the OpenAI-compatible server in-process and stream a completion over SSE.

    python examples/02_stream_client.py

Against a standalone server (`tinyserve serve --port 8000`) the same request is:

    curl -N http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
         -d '{"prompt": "ROMEO:\\n", "max_tokens": 100, "stream": true, "seed": 7}'
"""

import json
import urllib.request

from tinyserve import EngineConfig, LLMEngine, load_pretrained
from tinyserve.server import Server

model, tok = load_pretrained("target")
srv = Server(LLMEngine(model, tok, EngineConfig()), port=0).start_background()

body = {"prompt": "ROMEO:\n", "max_tokens": 150, "temperature": 0.8, "seed": 7, "stream": True}
req = urllib.request.Request(f"http://127.0.0.1:{srv.port}/v1/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
print(body["prompt"], end="", flush=True)
chunks = 0
with urllib.request.urlopen(req) as resp:
    for raw in resp:                       # one SSE line at a time, as it arrives
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        data = line[len("data: "):]
        if data == "[DONE]":
            break
        choice = json.loads(data)["choices"][0]
        print(choice["text"], end="", flush=True)
        chunks += 1
print(f"\n\n[{chunks} SSE chunks, finish_reason={choice['finish_reason']}]")
srv.shutdown()
