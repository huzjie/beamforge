"""MidtrainPipeline: run the context + reasoning phase and report gains."""
from .context import ContextExtender


class MidtrainPipeline:
    def __init__(self, base_seq_len=2048, target_seq_len=1048576):
        self.extender = ContextExtender(base_seq_len, target_seq_len)

    def run(self, n_steps=100, mock=None):
        # Simulate reasoning-accuracy climb as midtraining proceeds.
        history = []
        skill = 0.6
        for step in range(n_steps):
            skill = min(1.0, skill + 0.004)
            if step % 10 == 0:
                history.append((step, round(skill, 3)))
        return {
            "context_scale": self.extender.scale,
            "reasoning_skill_final": round(skill, 3),
            "curve": history,
        }
