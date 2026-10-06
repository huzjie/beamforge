import unittest

from beamforge.midtrain.pipeline import MidtrainPipeline
from beamforge.midtrain.context import ContextExtender


class TestMidtrain(unittest.TestCase):
    def test_context_scale(self):
        ext = ContextExtender(base_seq_len=2048, target_seq_len=1048576)
        self.assertEqual(ext.scale, 512.0)

    def test_pipeline_runs(self):
        r = MidtrainPipeline().run(n_steps=50)
        self.assertGreater(r["reasoning_skill_final"], 0.6)


if __name__ == "__main__":
    unittest.main()
