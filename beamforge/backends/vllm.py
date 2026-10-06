"""vLLM backend (HTTP endpoint)."""
from .base import BaseBackend
from ..backends import register_backend


@register_backend("vllm")
class VLLMBackend(BaseBackend):
    name = "vllm"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.base_url = (self.cfg.base_url or "http://localhost:8000/v1").rstrip("/")
        self.model = self.cfg.model or "beamforge"

    def generate(self, prompt, **kw):
        import json
        import urllib.request
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "max_tokens": kw.get("max_tokens", self.cfg.max_tokens),
            "temperature": kw.get("temperature", self.cfg.temperature),
        }).encode()
        req = urllib.request.Request(f"{self.base_url}/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode())
            return {"answer": data["choices"][0]["text"], "correct": None}
        except Exception as e:  # noqa: BLE001
            return {"answer": "", "error": f"vllm unreachable: {e}"}
