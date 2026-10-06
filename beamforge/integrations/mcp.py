"""Minimal MCP (Model Context Protocol) tool server definition.

Exposes the beamforge pipeline as a set of JSON-schema-described tools so an MCP
host could call `beamforge_train`, `beamforge_bench`, etc.
"""
import json


class MCPServer:
    TOOLS = [
        {"name": "beamforge_train",
         "description": "Run the pretrain -> midtrain -> RL training pipeline",
         "inputSchema": {"type": "object", "properties": {
             "steps": {"type": "integer", "default": 50}}}},
        {"name": "beamforge_bench",
         "description": "Run all four benchmarks (throughput / fault / scaling / sandbox)",
         "inputSchema": {"type": "object", "properties": {
             "n_faults": {"type": "integer", "default": 71}}}},
        {"name": "beamforge_answer",
         "description": "Ask the engine a question",
         "inputSchema": {"type": "object", "properties": {
             "prompt": {"type": "string"}}, "required": ["prompt"]}},
    ]

    def list_tools(self):
        return self.TOOLS

    def call(self, name, arguments):
        from ..train.factory import TrainingFactory
        from ..bench import run_all
        from ..model.engine import BeamEngine
        if name == "beamforge_train":
            return TrainingFactory().run_all(pretrain_steps=arguments.get("steps", 50))
        if name == "beamforge_bench":
            return run_all(n_faults=arguments.get("n_faults", 71))
        if name == "beamforge_answer":
            return BeamEngine().answer(arguments["prompt"])
        raise KeyError(name)

    def manifest(self):
        return json.dumps({"tools": self.TOOLS}, ensure_ascii=False)
