import unittest

from beamforge.train.rl_runner import RLRunner


class TestRL(unittest.TestCase):
    def test_skill_climbs(self):
        r = RLRunner(skill=0.5).run(n_prompts=20, k=4)
        self.assertGreater(r["skill_final"], r["skill_initial"])


if __name__ == "__main__":
    unittest.main()
