"""Throughput benchmark: MoE sparse routing vs dense-equivalent.

Reports communication cost per token (top_k * dim) which is independent of the
total number of experts -- the core reason frontier MoE stays fast.
"""
from ..bench import register_benchmark


@register_benchmark("throughput")
def run_throughput(dim=64, top_k=2):
    rows = []
    for n_experts in (8, 32, 128, 384, 512):
        comm = top_k * dim
        dense_comm = n_experts * dim  # a dense layer would touch every expert
        rows.append({
            "n_experts": n_experts,
            "top_k": top_k,
            "sparse_ratio": n_experts / top_k,
            "comm_per_token": comm,
            "dense_comm": dense_comm,
            "speedup_vs_dense": round(dense_comm / comm, 1),
        })
    return {"rows": rows, "note": "comm scales with top_k, not n_experts"}
