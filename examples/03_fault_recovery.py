"""Example script (run from repo root or examples/)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamforge.fault.orchestrator import FaultTolerantOrchestrator

orch = FaultTolerantOrchestrator(n_faults=71, seed=7)

def train_step(step, state):
    state["skill"] = min(1.0, state.get("skill", 0.5) + 0.001)
    return state

orch.run(total_steps=600, train_step_fn=train_step)
print(orch.report())
