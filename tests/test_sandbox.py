import unittest

from beamforge.sandbox.env import SandboxEnv
from beamforge.sandbox.factory import SandboxFactory


class TestSandbox(unittest.TestCase):
    def test_env_deterministic(self):
        e1 = SandboxEnv("code", 1)
        e2 = SandboxEnv("code", 1)
        self.assertEqual(e1.observe(), e2.observe())

    def test_factory_throughput(self):
        def policy(obs):
            return max(range(4), key=lambda a: obs[a % len(obs)])
        res = SandboxFactory(capacity=16).run_episodes(50, policy)
        self.assertGreater(res["episodes"], 0)
        self.assertGreater(res["throughput_eps_per_sec"], 0)


if __name__ == "__main__":
    unittest.main()
