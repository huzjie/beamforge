"""Top-K gating router.

Reusable trick: the router emits logits per expert, picks top-K, and gates via a
softmax over the *selected* experts only. Keep the router tiny (dim -> n_experts)
so the marginal cost of adding experts is just the dispatch, not the gate.
"""
import math


class TopKRouter:
    def __init__(self, dim, n_experts, top_k, seed=0):
        import random
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = top_k
        r = random.Random(seed)
        bound = 1.0 / math.sqrt(dim)
        self.w = [[r.uniform(-bound, bound) for _ in range(dim)] for _ in range(n_experts)]

    def logits(self, x):
        # x: list[float] of dim
        return [sum(x[k] * wj[k] for k in range(self.dim)) for wj in self.w]

    def route(self, x):
        lg = self.logits(x)
        order = sorted(range(self.n_experts), key=lambda j: -lg[j])
        chosen = order[:self.top_k]
        # softmax over chosen only
        m = max(lg[j] for j in chosen)
        exps = {j: math.exp(lg[j] - m) for j in chosen}
        s = sum(exps.values())
        gates = {j: exps[j] / s for j in chosen}
        return chosen, gates
