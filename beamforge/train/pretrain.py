"""Pretraining loop (mock): loss decreases, skill climbs."""
from ..backends.mock import MockBackend, world_answer


class PretrainRunner:
    def __init__(self, backend=None):
        self.backend = backend or MockBackend()

    def run(self, n_steps=50, lr=0.1):
        history = []
        loss = 1.0
        correct = 0
        for step in range(n_steps):
            prompt = f"pretrain:{step}"
            truth = world_answer(prompt)
            r = self.backend.generate(prompt)
            self.backend.train_step(r["correct"])
            loss = loss * 0.92 + (0.0 if r["correct"] else 0.3) * 0.08
            if r["correct"]:
                correct += 1
            history.append((step, round(self.backend.skill, 3), round(loss, 3)))
        return {
            "final_skill": round(self.backend.skill, 3),
            "final_loss": round(loss, 4),
            "accuracy": round(correct / n_steps, 3),
            "curve": history,
        }
