import unittest

from beamforge.config import Config, load_config


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = Config()
        self.assertEqual(c.backend.name, "mock")
        self.assertEqual(c.moe.top_k, 2)

    def test_from_dict(self):
        c = Config.from_dict({"backend": {"name": "cpu"}, "moe": {"top_k": 4}})
        self.assertEqual(c.backend.name, "cpu")
        self.assertEqual(c.moe.top_k, 4)


if __name__ == "__main__":
    unittest.main()
