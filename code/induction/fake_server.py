#!/usr/bin/env python3
"""
fake_server.py — an OpenAI-compatible stand-in for vLLM, for testing the pipeline without a GPU.

The "model" tries a few canned transforms in turn; when the executor reply says all demos passed
it calls commit_solution. Solves trivial tasks only — that is the point: it exercises the loop,
the server handshake, the verifier and the sweep summary end to end in seconds.

  python fake_server.py --port 8000
"""
import argparse
import http.server
import json
import time

PROGRAMS = [
    "import numpy as np\ndef transform(grid):\n    g = np.array(grid)\n    return np.kron((g != 0).astype(int), g).tolist()\n",
    "import numpy as np\ndef transform(grid):\n    return np.rot90(np.array(grid)).tolist()\n",
    "import numpy as np\ndef transform(grid):\n    return np.array(grid)[::-1].tolist()\n",
    "import numpy as np\ndef transform(grid):\n    g = np.array(grid)\n    return np.hstack([g, g[:, ::-1]]).tolist()\n",
    "def transform(grid):\n    return grid\n",
]


class H(http.server.BaseHTTPRequestHandler):
    delay = 0.0

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        out = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._send({"object": "list", "data": [{"id": "fake", "object": "model"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = body["messages"]
        n_assistant = sum(m["role"] == "assistant" for m in msgs)
        last_tool = next((m["content"] for m in reversed(msgs) if m["role"] == "tool"), "")
        time.sleep(self.delay)
        if "PASSED!" in last_tool:
            call = {"name": "commit_solution", "arguments": json.dumps({"step": None})}
        else:
            code = PROGRAMS[min(n_assistant, len(PROGRAMS) - 1)]
            call = {"name": "arc_python_executor", "arguments": json.dumps({"code": f"```python\n{code}```"})}
        msg = {"role": "assistant", "content": "Looking at the examples, I will test a hypothesis.",
               "reasoning_content": "fake reasoning",
               "tool_calls": [{"id": f"call_{n_assistant}", "type": "function", "function": call}]}
        self._send({"id": "fake", "object": "chat.completion", "model": body.get("model", "fake"),
                    "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 4000 + 1500 * n_assistant, "completion_tokens": 900}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds per reply, to mimic generation time")
    a = ap.parse_args()
    H.delay = a.delay
    print(f"fake model server on http://127.0.0.1:{a.port}/v1", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
