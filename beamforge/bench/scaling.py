"""Sparse-scaling benchmark: total vs active parameters."""
from ..bench import register_benchmark
from ..moe.sparse import SparseAccountant


@register_benchmark("scaling")
def run_scaling(dim=64, hidden=64, top_k=2):
    expert_params = hidden * dim * 3  # W1 + W3 + W2
    rows = []
    for n_experts in (8, 32, 128, 384, 512):
        acc = SparseAccountant(n_experts, top_k, expert_params, dim)
        rows.append(acc.report())
    return {"rows": rows, "expert_params": expert_params}
