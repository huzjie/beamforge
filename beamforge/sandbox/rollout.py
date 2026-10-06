"""Rollout collection: turn sandbox trajectories into (state, action, reward)."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Rollout:
    kind: str
    env_id: int
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    @property
    def total_reward(self):
        return sum(self.rewards)


class RolloutCollector:
    def __init__(self, policy_fn):
        # policy_fn(observation) -> action (int)
        self.policy_fn = policy_fn

    def collect(self, env, n_episodes=1):
        rollouts = []
        for ep in range(n_episodes):
            env.reset()
            r = Rollout(kind=env.kind, env_id=env.env_id)
            for _ in range(env.max_steps):
                obs = env.observe()
                action = self.policy_fn(obs)
                reward, done = env.step_env(action)
                r.actions.append(action)
                r.rewards.append(reward)
                if done:
                    break
            rollouts.append(r)
        return rollouts
