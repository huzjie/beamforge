"""Deterministic fault injector for training a *recovery policy*.

Rather than waiting for real faults, we inject a scripted stream of faults whose
type and timing are deterministic (md5-seeded), so a run is reproducible and the
MTTR benchmark is stable across runs.
"""
from .taxonomy import ErrorClass


class FaultInjector:
    def __init__(self, n_faults=71, seed=0):
        self.n_faults = n_faults
        self.seed = seed
        self.stream = self._build_stream()

    def _build_stream(self):
        from ..utils.stable import stable_choice, stable_ints
        kinds = [ErrorClass.HARDWARE, ErrorClass.NETWORK, ErrorClass.OOM,
                 ErrorClass.DATA, ErrorClass.SOFTWARE]
        stream = []
        for i in range(self.n_faults):
            kind = stable_choice(f"fault:{self.seed}:{i}:kind", kinds)
            # step at which the fault fires
            step = stable_ints(f"fault:{self.seed}:{i}:step", 1, lo=1, hi=500)[0]
            stream.append({"id": i, "kind": kind, "step": step})
        stream.sort(key=lambda f: f["step"])
        return stream

    def faults_at(self, step):
        return [f for f in self.stream if f["step"] == step]
