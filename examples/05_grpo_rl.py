"""Example script (run from repo root or examples/)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamforge.train.rl_runner import RLRunner

r = RLRunner(skill=0.5).run(n_prompts=20, k=4)
print("skill:", r["skill_initial"], "->", r["skill_final"])
