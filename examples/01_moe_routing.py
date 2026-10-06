"""Example script (run from repo root or examples/)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamforge.moe.layer import MoELayer

moe = MoELayer(dim=32, hidden=32, n_experts=16, top_k=2, seed=42)
x = [0.1 * i for i in range(32)]
out, chosen, gates = moe.forward(x)
print("chosen experts:", chosen)
print("gates:", {k: round(v, 3) for k, v in gates.items()})
print("sparse_ratio:", moe.sparse_ratio)
print("total_params:", moe.total_params(), "active_params:", moe.active_params())
