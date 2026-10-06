import unittest

from beamforge.integrations.langchain import BeamForgeLLM
from beamforge.integrations.mcp import MCPServer
from beamforge.integrations.agent_loop import AgentLoop


class TestIntegrations(unittest.TestCase):
    def test_llm_wrapper(self):
        llm = BeamForgeLLM()
        self.assertEqual(llm._llm_type, "beamforge")
        self.assertIn("answer", llm.engine.answer("hi"))

    def test_mcp_manifest(self):
        s = MCPServer()
        self.assertGreaterEqual(len(s.list_tools()), 3)
        self.assertIn("tools", s.manifest())

    def test_agent_loop(self):
        res = AgentLoop(max_steps=3).run("test")
        self.assertGreaterEqual(res["steps"], 1)


if __name__ == "__main__":
    unittest.main()
