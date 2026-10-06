"""Example script (run from repo root or examples/)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamforge.model.engine import BeamEngine

e = BeamEngine()
print("backend:", e.backend.name, "skill:", e.skill())
print("answer:", e.answer("what is 2+2?"))
