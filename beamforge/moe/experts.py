"""SwiGLU expert (a two-layer FFN with a gated first stage).

Shape contract (kept consistent everywhere, see the pitfall note below):
  W1, W3 : (hidden, dim)   -- gate / up projections
  W2     : (dim, hidden)   -- down projection
Forward:  h = swiglu(x @ W1^T, x @ W3^T); out = h @ W2^T  (dim -> hidden -> dim)
"""
import math


def _swiglu(g, u):
    return [a / (1.0 + math.exp(-a)) * b for a, b in zip(g, u)]


class SwiGLUExpert:
    def __init__(self, dim, hidden, seed=0):
        import random
        self.dim = dim
        self.hidden = hidden
        r = random.Random(seed)
        b = 1.0 / math.sqrt(dim)
        self.w1 = [[r.uniform(-b, b) for _ in range(dim)] for _ in range(hidden)]  # gate
        self.w3 = [[r.uniform(-b, b) for _ in range(dim)] for _ in range(hidden)]  # up
        self.w2 = [[r.uniform(-b, b) for _ in range(hidden)] for _ in range(dim)]  # down

    def forward(self, x):
        # x: list[float] of dim
        gate = [sum(x[k] * self.w1[j][k] for k in range(self.dim)) for j in range(self.hidden)]
        up = [sum(x[k] * self.w3[j][k] for k in range(self.dim)) for j in range(self.hidden)]
        h = _swiglu(gate, up)
        out = [sum(h[j] * self.w2[i][j] for j in range(self.hidden)) for i in range(self.dim)]
        return out

    def param_count(self):
        return self.hidden * self.dim * 3
