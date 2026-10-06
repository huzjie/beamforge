import unittest

from beamforge.moe.layer import MoELayer
from beamforge.moe.load_balance import aux_load_balance_loss


class TestMoE(unittest.TestCase):
    def test_forward_shape(self):
        moe = MoELayer(dim=16, hidden=16, n_experts=8, top_k=2)
        out, chosen, gates = moe.forward([0.1] * 16)
        self.assertEqual(len(out), 16)
        self.assertEqual(len(chosen), 2)

    def test_sparse_ratio(self):
        moe = MoELayer(dim=16, hidden=16, n_experts=8, top_k=2)
        self.assertEqual(moe.sparse_ratio, 4.0)

    def test_load_balance_loss(self):
        # perfectly balanced -> loss ~0
        self.assertAlmostEqual(aux_load_balance_loss([0, 1, 2, 3], 4), 0.0, places=6)
        # imbalanced -> positive
        self.assertGreater(aux_load_balance_loss([0, 0, 0, 0], 4), 0.0)


if __name__ == "__main__":
    unittest.main()
