"""Midtrain runner: context extension + reasoning climb."""
from ..midtrain.pipeline import MidtrainPipeline


class MidtrainRunner:
    def __init__(self, base_seq_len=2048, target_seq_len=1048576):
        self.pipeline = MidtrainPipeline(base_seq_len, target_seq_len)

    def run(self, n_steps=100):
        return self.pipeline.run(n_steps)
