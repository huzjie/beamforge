"""SandboxFactory: the high-level "AI factory" entrypoint.

Combines pool + scheduler + collector into a single object you can drive to
measure sandbox throughput (sandboxes / second) and elastic scaling behaviour.
"""
import time

from .env import SandboxEnv
from .pool import SandboxPool
from .scheduler import ElasticScheduler
from .rollout import RolloutCollector


class SandboxFactory:
    def __init__(self, kinds=("code", "web", "agent"), capacity=64):
        self.kinds = kinds
        self.pool = SandboxPool(kinds, capacity=capacity)
        self.scheduler = ElasticScheduler(self.pool)
        self.created = 0

    def run_episodes(self, n_episodes, policy_fn):
        collector = RolloutCollector(policy_fn)
        start = time.time()
        total = 0
        for ep in range(n_episodes):
            kind = self.kinds[ep % len(self.kinds)]
            env = SandboxEnv(kind, ep)
            rollouts = collector.collect(env, n_episodes=1)
            total += len(rollouts)
            self.scheduler.tick(queued=n_episodes - ep - 1)
        elapsed = max(time.time() - start, 1e-6)
        return {
            "episodes": total,
            "elapsed_seconds": round(elapsed, 3),
            "throughput_eps_per_sec": round(total / elapsed, 1),
            "pool_capacity": self.pool.capacity,
            "total_created": self.pool.total_created,
        }
