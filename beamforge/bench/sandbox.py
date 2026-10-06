"""Sandbox-factory throughput benchmark (episodes / second)."""
from ..bench import register_benchmark
from ..sandbox.factory import SandboxFactory


@register_benchmark("sandbox")
def run_sandbox(n_episodes=200, capacity=64):
    def policy(obs):
        # greedy-ish deterministic policy
        return max(range(4), key=lambda a: obs[a % len(obs)])
    factory = SandboxFactory(capacity=capacity)
    return factory.run_episodes(n_episodes, policy)
