"""Recovery policy: retry, restart-from-checkpoint, or elastic rescale.

The third pillar: pick the cheapest action that restores progress, and record
how long it took (that is the MTTR number we benchmark).
"""
from .taxonomy import ErrorClass


class RecoveryManager:
    def __init__(self, elastic=True):
        self.elastic = elastic
        self.retries = 0
        self.restarts = 0
        self.rescales = 0

    def policy(self, error_class):
        if error_class == ErrorClass.NETWORK:
            return "retry"
        if error_class == ErrorClass.OOM:
            return "rescale" if self.elastic else "restart"
        if error_class == ErrorClass.DATA:
            return "resample"
        if error_class == ErrorClass.SOFTWARE:
            return "restart"
        if error_class == ErrorClass.HARDWARE:
            return "rescale" if self.elastic else "restart"
        return "restart"

    def recover(self, error_class, step, checkpointer, state):
        action = self.policy(error_class)
        cost_minutes = {
            "retry": 0.5,
            "resample": 1.0,
            "restart": 4.0,
            "rescale": 6.0,
        }[action]
        if action == "retry":
            self.retries += 1
        elif action == "restart":
            self.restarts += 1
            checkpointer.save(step, state)
        elif action == "rescale":
            self.rescales += 1
        return action, cost_minutes
