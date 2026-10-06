"""Example script (run from repo root or examples/)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamforge.sandbox.factory import SandboxFactory

def policy(obs):
    return max(range(4), key=lambda a: obs[a % len(obs)])

res = SandboxFactory(capacity=64).run_episodes(n_episodes=200, policy_fn=policy)
print(res)
