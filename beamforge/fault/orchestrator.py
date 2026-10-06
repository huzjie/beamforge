"""Fault-tolerant training orchestrator: the "AI factory" loop.

Ties together injector + taxonomy + checkpointer + recovery + MTTR into a single
training loop that keeps making progress while faults are being injected, and
reports the recovery statistics at the end.
"""
from .taxonomy import ErrorTaxonomy
from .injector import FaultInjector
from .checkpoint import Checkpointer
from .recovery import RecoveryManager
from .mttr import MTTRTracker


class FaultTolerantOrchestrator:
    def __init__(self, n_faults=71, checkpoint_every=50, elastic=True, seed=0,
                 checkpoint_dir=".beamforge_ckpt"):
        self.injector = FaultInjector(n_faults=n_faults, seed=seed)
        self.checkpointer = Checkpointer(checkpoint_dir, every=checkpoint_every)
        self.recovery = RecoveryManager(elastic=elastic)
        self.mttr = MTTRTracker()
        self.errors = []
        self.completed_steps = 0

    def run(self, total_steps, train_step_fn):
        """train_step_fn(step, state) -> new_state (a small dict)."""
        state = {"skill": 0.5}
        step = 0
        while step < total_steps:
            faults = self.injector.faults_at(step)
            for f in faults:
                klass = f["kind"]
                action, minutes = self.recovery.recover(klass, step, self.checkpointer, state)
                self.mttr.record(minutes)
                self.errors.append({"id": f["id"], "class": klass.value,
                                    "action": action, "minutes": minutes})
                # elastic rescale gives a small boost to represent added capacity
                if action == "rescale":
                    state["skill"] = min(1.0, state.get("skill", 0.5) + 0.005)
            state = train_step_fn(step, state)
            if self.checkpointer.should_checkpoint(step + 1):
                self.checkpointer.save(step + 1, state)
            step += 1
            self.completed_steps = step
        return state

    def report(self):
        from .taxonomy import ErrorTaxonomy
        return {
            "completed_steps": self.completed_steps,
            "n_errors": len(self.errors),
            "taxonomy": ErrorTaxonomy.summary([e["class"] for e in self.errors]),
            "mttr": self.mttr.report(),
            "recovery_actions": {
                "retries": self.recovery.retries,
                "restarts": self.recovery.restarts,
                "rescales": self.recovery.rescales,
            },
        }
