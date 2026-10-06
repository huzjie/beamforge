"""Group Relative Policy Optimization (GRPO).

For each prompt, sample K responses, score them, and use the *group-relative*
advantage (score - group_mean) / group_std to update the policy. This removes
the need for a separate value/critic network.

The scalar-skill policy is updated with a REINFORCE-style gradient:
`grad = mean_i( advantage_i * a_i )` where `a_i = +1` for a correct response and
`-1` for a wrong one. Because correctness drives the reward, correct responses
carry positive advantage and wrong ones negative, so the gradient is positive
and skill climbs monotonically toward 1.0.
"""
import math


class GRPOTrainer:
    def __init__(self, policy, reward_model, lr=0.1, kl_coef=0.01):
        self.policy = policy
        self.reward = reward_model
        self.lr = lr
        self.kl_coef = kl_coef

    def step(self, prompt, k=4, ref_length=10):
        responses = [self.policy.sample(prompt, i) for i in range(k)]
        scored = [self.reward.score(r["correct"], r["length"], ref_length) for r in responses]
        mean = sum(scored) / len(scored)
        std = math.sqrt(sum((s - mean) ** 2 for s in scored) / len(scored)) + 1e-6
        advantages = [(s - mean) / std for s in scored]
        # REINFORCE on the scalar skill: +1 for correct, -1 for wrong
        grad = sum(a * (1.0 if r["correct"] else -1.0)
                   for a, r in zip(advantages, responses)) / k
        self.policy.update_grad(grad)
        return {
            "scores": scored,
            "mean_score": round(mean, 4),
            "skill": round(self.policy.skill, 4),
        }
