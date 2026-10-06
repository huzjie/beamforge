"""Deterministic, trainable mock backend.

This is the reference backend: no network, no model weights, but it *learns*.
The policy's "skill" is a scalar; answer correctness is `skill` (gated by a
deterministic world model) plus noise, so:

- training has a real, monotone effect (skill 0.5 -> 1.0),
- benchmarks are reproducible.

Two iron rules baked in (see the framework notes):
  1. skill must *participate in scoring*, otherwise training does nothing.
  2. the training delta must be *positive*, otherwise low skill collapses to 0.
"""
from .base import BaseBackend
from ..backends import register_backend
from ..utils.stable import stable_float, stable_choice


def world_answer(prompt):
    # deterministic ground-truth answer for a prompt (what a perfect model says)
    return stable_choice(f"truth:{prompt}", ["A", "B", "C", "D"])


@register_backend("mock")
class MockBackend(BaseBackend):
    name = "mock"

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.skill = 0.5
        self.history = [0.5]

    def generate(self, prompt, **kw):
        truth = world_answer(prompt)
        options = ["A", "B", "C", "D"]
        if (self.skill + stable_float(f"ans:{prompt}", -0.5, 0.5)) > 0.5:
            answer = truth
        else:
            answer = stable_choice(f"wrong:{prompt}", [o for o in options if o != truth])
        return {"answer": answer, "truth": truth, "correct": answer == truth, "skill": self.skill}

    def train_step(self, is_correct):
        # monotone-positive delta: correct gives a big push, wrong a small one,
        # so skill always drifts toward 1.0 (never collapses to 0).
        self.skill = min(1.0, self.skill + (0.08 if is_correct else 0.02))
        self.history.append(self.skill)
        return self.skill
