# Changelog

## 0.1.0a1 — First release

Embodify gives your agent a body: an MCP server and a set of skills that let
Codex, Claude Code and other agents control robots.

**MCP server (`embodify-mcp`)**

- Tools: `get_session_info`, `list_tasks`, `reset_task`, `observe`,
  `move_relative`, `set_gripper`, `stop_episode`, and `control_arms` for
  coordinated two-arm motion.
- Backends: one- and two-arm Fake, LIBERO with all 130 tasks, and RoboDojo with
  dual ARX X5 arms on all 54 simulation tasks, chosen by the agent or fixed by
  the operator, each with its native episode limit unless overridden.
- Remote backends over SSH or TCP, with heartbeats for slow links and
  reconnection through `reset_task`.
- Live monitor and replay: watch running episodes in a browser, or replay past
  runs frame by frame, in English or Chinese.
- MCP over stdio, protocol versions 2024-11-05 to 2025-11-25.
- Commands: `embodify-mcp`, `embodify-mcp-backend`, `embodify-mcp-monitor`,
  `embodify-mcp-preflight` and `embodify-mcp-smoke`.

**Skills pack (`embodify-skills`)**

- `embodied-control`, `robot-profile` and `task-experience`.
- Coming soon: `object-segmentation` and `depth-ranging`.

**Installation**

- Plugin marketplace in `.claude-plugin/` for Claude Code and Codex: install the
  skills with `claude plugin install embodify-skills@embodify` or
  `codex plugin add embodify-skills@embodify`, and register the MCP server as
  `embodify`.
- One-line setup: agents install Embodify for themselves by following
  `docs/agent-setup.md`.
