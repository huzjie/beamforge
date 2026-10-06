"""Sparse-accounting: track total vs active params and comm cost.

The key engineering insight behind Beam-style frontier MoE: communication and
FLOPs scale with the *active* parameters, not the total. This module turns that
into concrete numbers for the scaling benchmark.
"""


class SparseAccountant:
    def __init__(self, n_experts, top_k, expert_params, dim):
        self.n_experts = n_experts
        self.top_k = top_k
        self.expert_params = expert_params
        self.dim = dim

    @property
    def total(self):
        return self.n_experts * self.expert_params

    @property
    def active(self):
        return self.top_k * self.expert_params

    @property
    def sparse_ratio(self):
        return self.total / max(self.active, 1)

    def comm_per_token(self):
        # tokens dispatched to experts: seq_len x top_k x dim; here per-token basis
        return self.top_k * self.dim

    def report(self):
        return {
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "total_params": self.total,
            "active_params": self.active,
            "sparse_ratio": self.sparse_ratio,
            "comm_per_token": self.comm_per_token(),
        }
