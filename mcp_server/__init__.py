"""MCP(stdio)server 包:把已有 service 能力暴露给外部 Agent。

并列新增,不接入现有主链路:
- 不替换 agents/agentic/agentic_trial_booking.py 的 tool / bind_tools 路径;
- 不改 /chat/stream、不改 api/*;
- 依赖方向单向:mcp_server/ → services/ / config/ / db/,禁止被 agents/ api/ 反向依赖。
"""

from mcp_server.server import create_server, run_stdio

__all__ = ["create_server", "run_stdio"]
