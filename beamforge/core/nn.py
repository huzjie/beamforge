"""Parameterized layers built on Tensor."""
import math

from .tensor import Tensor


class Parameter:
    def __init__(self, data):
        self.data = data if isinstance(data, Tensor) else Tensor(data, requires_grad=True)


class Linear:
    def __init__(self, in_features, out_features, bias=True, seed=None):
        import random
        r = random.Random(seed)
        bound = 1.0 / math.sqrt(in_features)
        self.weight = Parameter([[r.uniform(-bound, bound) for _ in range(in_features)]
                                for _ in range(out_features)])
        self.bias = Parameter([r.uniform(-bound, bound) for _ in range(out_features)]) if bias else None

    def __call__(self, x):
        # weight is (out, in); x is (rows, in); out = x @ W^T
        w = self.weight.data
        out = Tensor([[sum(x.data[i][k] * w.data[j][k] for k in range(x.cols))
                       for j in range(w.rows)] for i in range(x.rows)],
                     requires_grad=x.requires_grad, _children=(x,))
        if self.bias is not None:
            b = self.bias.data
            out = Tensor([[out.data[i][j] + b.data[0][j] for j in range(out.cols)]
                          for i in range(out.rows)], requires_grad=out.requires_grad, _children=(out,))
        return out


class Embedding:
    def __init__(self, num_embeddings, dim, seed=None):
        import random
        r = random.Random(seed)
        self.table = [[r.uniform(-0.1, 0.1) for _ in range(dim)] for _ in range(num_embeddings)]

    def __call__(self, ids):
        return Tensor([list(self.table[i % len(self.table)]) for i in ids])


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.dim = dim
        self.eps = eps

    def __call__(self, x):
        from .functional import layernorm
        return layernorm(x, self.eps)
