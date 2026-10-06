# 集成

- **LangChain**：`from beamforge.integrations.langchain import BeamForgeLLM`。
- **MCP**：`from beamforge.integrations.mcp import MCPServer`，`manifest()` 给出工具清单。
- **Agent loop**：`from beamforge.integrations.agent_loop import AgentLoop`。
- **OpenAI 客户端**：`beamforge serve` 起服务后用任意 OpenAI SDK 连 `/v1/chat/completions`。
