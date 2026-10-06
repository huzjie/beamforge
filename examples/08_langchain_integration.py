"""Example script (run from repo root or examples/)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamforge.integrations.langchain import BeamForgeLLM

llm = BeamForgeLLM()
print("llm_type:", llm._llm_type)
print("answer:", llm("what is the capital of France?"))
