"""Reward shaping: accuracy minus a length penalty.

Reusable decision: reward = accuracy - lambda * length_ratio. The length penalty
is what stops the model from rambling forever to boost its odds of a hit.
"""
from ..utils.stable import stable_float


def shaped_reward(is_correct, length, ref_length, lam=0.5):
    ratio = length / max(ref_length, 1)
    return (1.0 if is_correct else 0.0) - lam * ratio


class RewardModel:
    def __init__(self, lam=0.5):
        self.lam = lam

    def score(self, is_correct, length, ref_length):
        return shaped_reward(is_correct, length, ref_length, self.lam)
