"""OpenAI-compatible backend (httpx-free; uses stdlib urllib)."""
from .base import BaseBackend
from ..backends import register_backend


@register_backend("openai")
class OpenAIBackend(BaseBackend):
    name = "openai"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.base_url = (self.cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = self.cfg.api_key or ""
        self.model = self.cfg.model or "gpt-4o-mini"

    def generate(self, prompt, **kw):
        if not self.api_key:
            return {"answer": "", "error": "openai backend requires api_key"}
        import json
        import urllib.request
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kw.get("max_tokens", self.cfg.max_tokens),
            "temperature": kw.get("temperature", self.cfg.temperature),
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode())
        answer = data["choices"][0]["message"]["content"]
        return {"answer": answer, "correct": None}
