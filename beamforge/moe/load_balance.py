"""Load-balancing auxiliary loss (router-z / switch loss).

Encourages tokens to spread evenly across experts. Without it, a few experts get
all the traffic and the others sit idle -- the classic MoE failure mode.
"""
import math


def load_fraction(counts, total):
    if total == 0:
        return [0.0] * len(counts)
    return [c / total for c in counts]


def aux_load_balance_loss(assignments, n_experts):
    """assignments: list of chosen expert ids (one per token, already top-K)."""
    n = len(assignments)
    if n == 0:
        return 0.0
    counts = [0] * n_experts
    for j in assignments:
        counts[j] += 1
    f = load_fraction(counts, n)
    # sum_j f_j^2 : 1.0 means totally imbalanced, 1/n_experts means perfect
    return sum(x * x for x in f) - 1.0 / n_experts


def ideal_loss(n_experts):
    return 1.0 / n_experts - 1.0 / n_experts
