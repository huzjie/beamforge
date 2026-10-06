"""A tiny pure-Python tensor with autodiff.

Supports the op subset the framework needs (matmul, add, mul, relu, sigmoid,
softmax, layernorm, embeddings) with reverse-mode gradients via a tape. It is a
*reference* core -- not fast -- but zero-dependency and fully deterministic,
which is exactly what the mock backend and benchmarks require.
"""
import math


class Tensor:
    __slots__ = ("data", "requires_grad", "grad", "_prev", "_backward", "_op")

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        if isinstance(data, Tensor):
            raise TypeError("Tensor wrapping Tensor")
        if data and not isinstance(data[0], (list, tuple)):
            data = [list(data)]
        self.data = [[float(x) for x in row] for row in data]
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op

    @property
    def shape(self):
        return (len(self.data), len(self.data[0]) if self.data else 0)

    @property
    def rows(self):
        return self.shape[0]

    @property
    def cols(self):
        return self.shape[1]

    def item(self):
        return self.data[0][0]

    def _ensure_grad(self):
        if self.grad is None:
            self.grad = [[0.0] * self.cols for _ in range(self.rows)]

    def zero_grad(self):
        self.grad = [[0.0] * self.cols for _ in range(self.rows)]

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor([[other]] * self.rows) if self.rows > 0 else Tensor([[other]])
        if isinstance(other, Tensor):
            o = other if other.rows == self.rows else _broadcast_rows(other, self.rows)
        else:
            o = other
        out = Tensor([[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(self.data, o.data)],
                     requires_grad=self.requires_grad or o.requires_grad,
                     _children=(self, o), _op="add")

        def _b():
            if self.requires_grad:
                self._ensure_grad()
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.grad[i][j] += out.grad[i][j]
            if o.requires_grad:
                o._ensure_grad()
                for i in range(o.rows):
                    for j in range(o.cols):
                        o.grad[i][j] += out.grad[i][j]
        out._backward = _b
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor([[other]])
        out = Tensor([[a * b for a, b in zip(r1, r2)] for r1, r2 in zip(self.data, other.data)],
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="mul")

        def _b():
            if self.requires_grad:
                self._ensure_grad()
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.grad[i][j] += out.grad[i][j] * other.data[i][j]
            if other.requires_grad:
                other._ensure_grad()
                for i in range(other.rows):
                    for j in range(other.cols):
                        other.grad[i][j] += out.grad[i][j] * self.data[i][j]
        out._backward = _b
        return out

    def matmul(self, other):
        assert self.cols == other.rows, f"matmul shape {self.shape} x {other.shape}"
        out = Tensor([[sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                       for j in range(other.cols)] for i in range(self.rows)],
                     requires_grad=self.requires_grad or other.requires_grad,
                     _children=(self, other), _op="matmul")

        def _b():
            if self.requires_grad:
                self._ensure_grad()
                for i in range(self.rows):
                    for k in range(self.cols):
                        s = 0.0
                        for j in range(other.cols):
                            s += out.grad[i][j] * other.data[k][j]
                        self.grad[i][k] += s
            if other.requires_grad:
                other._ensure_grad()
                for k in range(other.rows):
                    for j in range(other.cols):
                        s = 0.0
                        for i in range(self.rows):
                            s += self.data[i][k] * out.grad[i][j]
                        other.grad[k][j] += s
        out._backward = _b
        return out

    def relu(self):
        out = Tensor([[max(0.0, x) for x in row] for row in self.data],
                     requires_grad=self.requires_grad, _children=(self,), _op="relu")

        def _b():
            if self.requires_grad:
                self._ensure_grad()
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.grad[i][j] += out.grad[i][j] if self.data[i][j] > 0 else 0.0
        out._backward = _b
        return out

    def sigmoid(self):
        out = Tensor([[1.0 / (1.0 + math.exp(-x)) for x in row] for row in self.data],
                     requires_grad=self.requires_grad, _children=(self,), _op="sigmoid")

        def _b():
            if self.requires_grad:
                self._ensure_grad()
                for i in range(self.rows):
                    for j in range(self.cols):
                        s = out.data[i][j]
                        self.grad[i][j] += out.grad[i][j] * s * (1 - s)
        out._backward = _b
        return out

    def tanh(self):
        out = Tensor([[math.tanh(x) for x in row] for row in self.data],
                     requires_grad=self.requires_grad, _children=(self,), _op="tanh")

        def _b():
            if self.requires_grad:
                self._ensure_grad()
                for i in range(self.rows):
                    for j in range(self.cols):
                        t = out.data[i][j]
                        self.grad[i][j] += out.grad[i][j] * (1 - t * t)
        out._backward = _b
        return out

    def sum(self):
        out = Tensor([[sum(sum(r) for r in self.data)]],
                     requires_grad=self.requires_grad, _children=(self,), _op="sum")

        def _b():
            if self.requires_grad:
                self._ensure_grad()
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.grad[i][j] += out.grad[0][0]
        out._backward = _b
        return out

    def mean(self):
        n = self.rows * self.cols
        out = Tensor([[sum(sum(r) for r in self.data) / n]],
                     requires_grad=self.requires_grad, _children=(self,), _op="mean")

        def _b():
            if self.requires_grad:
                self._ensure_grad()
                g = out.grad[0][0] / n
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.grad[i][j] += g
        out._backward = _b
        return out

    def softmax(self):
        out_rows = []
        for row in self.data:
            m = max(row)
            exps = [math.exp(x - m) for x in row]
            s = sum(exps)
            out_rows.append([e / s for e in exps])
        out = Tensor(out_rows, requires_grad=self.requires_grad, _children=(self,), _op="softmax")

        def _b():
            if self.requires_grad:
                self._ensure_grad()
                for i in range(self.rows):
                    for j in range(self.cols):
                        s = out.data[i][j]
                        acc = 0.0
                        for k in range(self.cols):
                            acc += out.grad[i][k] * s * ((1.0 if k == j else 0.0) - out.data[i][k])
                        self.grad[i][j] += acc
        out._backward = _b
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(t):
            if t not in visited:
                visited.add(t)
                for c in t._prev:
                    build(c)
                topo.append(t)
        build(self)
        self.grad = [[1.0] * self.cols for _ in range(self.rows)]
        for t in reversed(topo):
            t._backward()

    @staticmethod
    def zeros(rows, cols, requires_grad=False):
        return Tensor([[0.0] * cols for _ in range(rows)], requires_grad=requires_grad)

    @staticmethod
    def ones(rows, cols, requires_grad=False):
        return Tensor([[1.0] * cols for _ in range(rows)], requires_grad=requires_grad)

    @staticmethod
    def randn(rows, cols, seed=None, requires_grad=False):
        import random
        r = random.Random(seed) if seed is not None else random
        return Tensor([[r.gauss(0, 1) for _ in range(cols)] for _ in range(rows)],
                      requires_grad=requires_grad)

    def __repr__(self):
        return f"Tensor(shape={self.shape})"


def _broadcast_rows(t, rows):
    if t.rows == 1:
        return Tensor([list(t.data[0]) for _ in range(rows)], requires_grad=t.requires_grad)
    return t
