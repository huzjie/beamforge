"""Example script (run from repo root or examples/)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamforge.train.pretrain import PretrainRunner

r = PretrainRunner().run(n_steps=50)
print("final_skill:", r["final_skill"], "final_loss:", r["final_loss"], "acc:", r["accuracy"])
print("curve head/tail:", r["curve"][0], r["curve"][-1])
