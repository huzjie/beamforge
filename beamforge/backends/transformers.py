"""HuggingFace transformers backend (optional dependency)."""
from .base import BaseBackend
from ..backends import register_backend


@register_backend("transformers")
class TransformersBackend(BaseBackend):
    name = "transformers"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.model = self.cfg.model or "gpt2"
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            try:
                from transformers import pipeline  # noqa: F401
                self._pipe = pipeline("text-generation", model=self.model)
            except Exception as e:  # noqa: BLE001
                return f"transformers unavailable: {e}"
        return None

    def generate(self, prompt, **kw):
        err = self._ensure()
        if err:
            return {"answer": "", "error": err}
        out = self._pipe(prompt, max_new_tokens=kw.get("max_tokens", 32))
        return {"answer": out[0]["generated_text"], "correct": None}
