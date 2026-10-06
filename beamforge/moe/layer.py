"""A sparse MoE layer: route each token to top-K experts.

The layer exposes both the forward pass and the per-step load statistics, so the
benchmarks can report the sparse ratio and communication cost *without* ever
allocating a full frontier-size model (see the pitfall note about keeping dims
small in smoke tests).
"""
from .router import TopKRouter
from .experts import SwiGLUExpert


class MoELayer:
    def __init__(self, dim, hidden, n_experts, top_k, seed=0):
        self.dim = dim
        self.hidden = hidden
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = TopKRouter(dim, n_experts, top_k, seed=seed)
        self.experts = [SwiGLUExpert(dim, hidden, seed=seed + 1000 + i) for i in range(n_experts)]

    def forward(self, x):
        chosen, gates = self.router.route(x)
        out = [0.0] * self.dim
        for j in chosen:
            ej = self.experts[j].forward(x)
            g = gates[j]
            for i in range(self.dim):
                out[i] += g * ej[i]
        return out, chosen, gates

    @property
    def sparse_ratio(self):
        # total parameters / active parameters per token
        return self.n_experts / self.top_k

    def total_params(self):
        return sum(e.param_count() for e in self.experts)

    def active_params(self):
        return self.top_k * self.experts[0].param_count()
