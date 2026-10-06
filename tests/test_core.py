import unittest

from beamforge.core.tensor import Tensor


class TestTensor(unittest.TestCase):
    def test_add(self):
        a = Tensor([[1.0, 2.0]])
        b = Tensor([[3.0, 4.0]])
        out = a + b
        self.assertEqual(out.data, [[4.0, 6.0]])

    def test_matmul_shape(self):
        a = Tensor([[1.0, 2.0, 3.0]])
        b = Tensor([[1.0], [2.0], [3.0]])
        self.assertEqual((a.matmul(b)).shape, (1, 1))
        self.assertAlmostEqual(a.matmul(b).item(), 14.0)

    def test_softmax_sums_to_one(self):
        t = Tensor([[1.0, 2.0, 3.0]]).softmax()
        self.assertAlmostEqual(sum(t.data[0]), 1.0, places=6)

    def test_backward(self):
        x = Tensor([[2.0]], requires_grad=True)
        y = x * x
        y.backward()
        self.assertAlmostEqual(x.grad[0][0], 4.0)


if __name__ == "__main__":
    unittest.main()
