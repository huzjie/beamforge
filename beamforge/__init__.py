"""beamforge: sparse frontier-model training factory.

A zero-dependency, fully runnable reference implementation of the training
techniques behind Reflection AI's Beam (announced 2026-10-05): a 501B-parameter
Mixture-of-Experts model that routes each token through only ~23B parameters,
with a 1M-token context, pre-trained on 23.8T tokens, midtrained for reasoning,
and RL-tuned in 1.3 billion sandboxes on a self-healing "AI factory" cluster
(median recovery 8 minutes across 71 faults).

This repo distills its most reusable engineering ideas into one framework:

1. Frontier sparse MoE -- top-k routing, SwiGLU experts, load-balancing loss,
   ~21.8x sparse ratio (501B total / 23B active).
2. Fault-tolerant elastic orchestration -- error taxonomy, deterministic fault
   injection, checkpoint/resume, elastic scale-in/out, MTTR tracking.
3. Sandbox factory -- massively-parallel RL rollout sandboxes (code / web / agent).
4. Midtraining -- context-window extension + reasoning enhancement as a distinct phase.

Highlights
----------
- Zero-dependency pure-Python tensor core with autodiff over the op set
- Deterministic trainable mock backend (skill monotonically -> 1.0)
- Fault-tolerance benchmark (71 injected faults, median recovery in minutes)
- Sandbox throughput + MoE sparse-scaling + midtrain context benchmarks
- CLI + Docker/K8s/Helm/CI + prompts + recipes
"""
from .version import __version__
from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
