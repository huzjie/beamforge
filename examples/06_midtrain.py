"""Example script (run from repo root or examples/)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamforge.midtrain.pipeline import MidtrainPipeline

r = MidtrainPipeline().run(n_steps=100)
print("context_scale:", r["context_scale"], "reasoning_skill_final:", r["reasoning_skill_final"])
