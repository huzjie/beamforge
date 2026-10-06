"""A zero-dependency OpenAI-compatible HTTP server (stdlib http.server).

Exposes /health, /v1/models, and /v1/chat/completions so any OpenAI client can
talk to a BeamEngine. No Flask/FastAPI needed.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..model.engine import BeamEngine


class _Handler(BaseHTTPRequestHandler):
    engine = None

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "skill": self.engine.skill()})
        elif self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [
                {"id": "beamforge-mock", "object": "model", "owned_by": "beamforge"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            prompt = ""
            for m in body.get("messages", []):
                prompt += m.get("content", "")
            out = self.engine.answer(prompt)
            resp = {
                "id": "cmpl-beamforge", "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant",
                                                    "content": out["answer"]},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
            }
            self._send(200, resp)
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        pass


def create_server(engine=None, port=8899):
    handler = type("H", (_Handler,), {"engine": engine or BeamEngine()})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def serve(engine=None, port=8899):
    srv = create_server(engine, port)
    print(f"[serve] beamforge on http://127.0.0.1:{port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
