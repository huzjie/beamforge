"""CPU reference backend: runs the tiny transformer locally."""
from .base import BaseBackend
from ..backends import register_backend
from ..model.transformer import BeamTransformer


@register_backend("cpu")
class CpuBackend(BaseBackend):
    name = "cpu"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.model = BeamTransformer()

    def generate(self, prompt, **kw):
        ids = [ord(c) % 256 for c in (prompt or "hi")[:8]]
        logits = self.model.forward(ids)
        best = max(range(len(logits)), key=lambda i: logits[i])
        return {"answer": chr(best), "logits": logits[:8], "correct": None}
