"""A minimal observe-act loop powered by the engine."""
from ..model.engine import BeamEngine


class AgentLoop:
    def __init__(self, backend=None, max_steps=5):
        self.engine = BeamEngine(backend=backend)
        self.max_steps = max_steps

    def run(self, task):
        trace = []
        for step in range(self.max_steps):
            out = self.engine.answer(f"{task} (step {step})")
            trace.append(out)
            if out.get("correct"):
                break
        return {"task": task, "steps": len(trace), "trace": trace}
