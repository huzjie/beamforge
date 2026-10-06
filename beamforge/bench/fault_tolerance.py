"""Fault-tolerance benchmark: inject 71 faults, measure MTTR."""
from ..bench import register_benchmark
from ..fault.orchestrator import FaultTolerantOrchestrator


@register_benchmark("fault_tolerance")
def run_fault_tolerance(n_faults=71, total_steps=600, seed=0):
    orch = FaultTolerantOrchestrator(n_faults=n_faults, seed=seed)

    def train_step(step, state):
        state["skill"] = min(1.0, state.get("skill", 0.5) + 0.001)
        return state

    orch.run(total_steps=total_steps, train_step_fn=train_step)
    return orch.report()
