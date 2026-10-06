import unittest

from beamforge.fault.taxonomy import ErrorTaxonomy, ErrorClass
from beamforge.fault.orchestrator import FaultTolerantOrchestrator


class TestFault(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(ErrorTaxonomy.classify("NCCL timeout"), ErrorClass.NETWORK)
        self.assertEqual(ErrorTaxonomy.classify("CUDA out of memory"), ErrorClass.OOM)

    def test_orchestrator_completes(self):
        orch = FaultTolerantOrchestrator(n_faults=10, seed=0, checkpoint_dir=".ckpt_test")

        def train_step(step, state):
            state["skill"] = min(1.0, state.get("skill", 0.5) + 0.001)
            return state

        orch.run(total_steps=600, train_step_fn=train_step)
        rep = orch.report()
        self.assertEqual(rep["n_errors"], 10)
        self.assertGreater(rep["mttr"]["median_minutes"], 0)


if __name__ == "__main__":
    unittest.main()
