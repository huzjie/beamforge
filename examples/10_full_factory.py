"""Example script (run from repo root or examples/)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamforge.train.factory import TrainingFactory
from beamforge.bench import run_all

train = TrainingFactory().run_all(pretrain_steps=50, midtrain_steps=100, rl_prompts=20)
bench = run_all(n_faults=71)

print("pretrain skill ->", train["pretrain"]["final_skill"])
print("rl skill ->", train["rl"]["skill_final"])
print("fault MTTR ->", bench["fault_tolerance"]["mttr"]["median_minutes"], "min")
print("scaling sparse_ratio@512 ->", bench["scaling"]["rows"][-1]["sparse_ratio"])
