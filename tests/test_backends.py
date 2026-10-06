import unittest

from beamforge.backends import get_backend, list_backends
from beamforge.backends.mock import MockBackend


class TestBackends(unittest.TestCase):
    def test_list_backends(self):
        self.assertIn("mock", list_backends())
        self.assertIn("openai", list_backends())

    def test_mock_trainable(self):
        b = MockBackend()
        s0 = b.skill
        for _ in range(40):
            b.train_step(True)
        self.assertGreater(b.skill, s0)
        self.assertAlmostEqual(b.skill, 1.0, places=2)

    def test_get_backend(self):
        b = get_backend("mock")
        self.assertEqual(b.name, "mock")


if __name__ == "__main__":
    unittest.main()
