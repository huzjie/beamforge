"""Backend base class."""
from ..config import BackendConfig


class BaseBackend:
    name = "base"

    def __init__(self, cfg=None):
        self.cfg = cfg or BackendConfig()

    def generate(self, prompt, **kw):
        raise NotImplementedError

    def train_step(self, is_correct):
        raise NotImplementedError
