"""Functional ops built on Tensor."""
import math

from .tensor import Tensor


def relu(x):
    return x.relu()


def sigmoid(x):
    return x.sigmoid()


def tanh(x):
    return x.tanh()


def softmax(x):
    return x.softmax()


def gelu(x):
    out = Tensor([[0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3)))
                   for v in row] for row in x.data],
                 requires_grad=x.requires_grad, _children=(x,))
    out._backward = x._backward
    return out


def swiglu(gate, up):
    # gate * sigmoid(gate) * up
    return gate.sigmoid().__mul__(gate).__mul__(up)


def layernorm(x, eps=1e-5):
    out_rows = []
    for row in x.data:
        m = sum(row) / len(row)
        v = sum((a - m) ** 2 for a in row) / len(row)
        out_rows.append([(a - m) / math.sqrt(v + eps) for a in row])
    out = Tensor(out_rows, requires_grad=x.requires_grad, _children=(x,))
    out._backward = x._backward
    return out


def rmsnorm(x, eps=1e-6):
    out_rows = []
    for row in x.data:
        ms = sum(a * a for a in row) / len(row)
        out_rows.append([a / math.sqrt(ms + eps) for a in row])
    out = Tensor(out_rows, requires_grad=x.requires_grad, _children=(x,))
    out._backward = x._backward
    return out


def cross_entropy(logits, target_idx):
    probs = logits.softmax()
    return Tensor([[-math.log(max(probs.data[0][target_idx], 1e-12))]],
                  requires_grad=logits.requires_grad, _children=(logits,))
