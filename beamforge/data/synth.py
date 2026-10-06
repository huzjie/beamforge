"""Deterministic synthetic corpora (pretraining tokens + QA pairs).

Because the mock backend is deterministic, synthetic data is enough to exercise
the full pipeline end-to-end without downloading any real dataset.
"""
from ..utils.stable import stable_choice, stable_ints


class SynthTokens:
    def __init__(self, vocab=256, seq_len=32):
        self.vocab = vocab
        self.seq_len = seq_len

    def batch(self, n=16, seed=0):
        return [stable_ints(f"tokens:{seed}:{i}", self.seq_len, lo=0, hi=self.vocab - 1)
                for i in range(n)]


class SynthQA:
    def __init__(self, n=200):
        self.n = n

    def pairs(self):
        return [(f"question:{i}", stable_choice(f"qa:{i}", ["A", "B", "C", "D"]))
                for i in range(self.n)]
