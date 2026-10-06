"""BeamEngine: a thin facade over a backend that adds a tiny world model.

The engine is what the benchmarks and the serving layer talk to; it can be
driven by any backend (mock/cpu/openai/vllm/transformers).
"""
from ..backends import get_backend
from ..config import Config


class BeamEngine:
    def __init__(self, backend=None, config=None):
        self.config = config or Config()
        self.backend = backend or get_backend(self.config.backend.name, self.config.backend)

    def answer(self, prompt, **kw):
        return self.backend.generate(prompt, **kw)

    def skill(self):
        return getattr(self.backend, "skill", None)

    def train_step(self, is_correct):
        return self.backend.train_step(is_correct)
