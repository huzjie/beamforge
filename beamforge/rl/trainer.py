"""A deterministic trainable RL policy (the GRPO "head").

Correctness of a sampled response is decided by comparing the policy's scalar
`skill` against a deterministic per-sample threshold, so the relationship is
monotone: higher skill -> more likely correct -> higher expected reward. The
GRPO gradient then pushes skill monotonically toward 1.0.
"""
from ..utils.stable import stable_float, stable_ints


class RLPolicy:
    def __init__(self, skill=0.5, lr=0.1):
        self.skill = skill
        self.lr = lr
        self.history = [skill]

    def sample(self, prompt, idx):
        # correctness = skill beats a uniform threshold (monotone in skill)
        threshold = stable_float(f"rl:{prompt}:{idx}:thr", 0.0, 1.0)
        correct = self.skill > threshold
        length = stable_ints(f"rl:{prompt}:{idx}:len", 1, lo=3, hi=16)[0]
        return {"correct": correct, "length": length}

    def update_grad(self, grad):
        self.skill = min(1.0, max(0.0, self.skill + self.lr * grad))
        self.history.append(self.skill)
