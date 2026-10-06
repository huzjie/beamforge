"""RL runner: GRPO over a trainable policy."""
from ..rl.reward import RewardModel
from ..rl.grpo import GRPOTrainer
from ..rl.trainer import RLPolicy


class RLRunner:
    def __init__(self, skill=0.5, lr=0.1, lam=0.5):
        self.policy = RLPolicy(skill=skill, lr=lr)
        self.reward = RewardModel(lam=lam)
        self.grp = GRPOTrainer(self.policy, self.reward, lr=lr)

    def run(self, n_prompts=20, k=4):
        skills = []
        for i in range(n_prompts):
            res = self.grp.step(f"prompt:{i}", k=k)
            skills.append(res["skill"])
        return {
            "skill_initial": self.policy.history[0],
            "skill_final": round(skills[-1], 3),
            "curve": [round(s, 3) for s in skills],
        }
