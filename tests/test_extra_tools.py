"""Backend-provided tools (Backend.extra_tools / call_extra) forwarded through the MCP session."""

import json

import numpy as np

from embodify_mcp.backend.base import ExtraResult, McpToolError
from embodify_mcp.backend.fake import FakeBackend
from embodify_mcp.mcp import McpSession

PING = {
    "name": "ping_body",
    "description": "For tests: echoes its argument and returns one image.",
    "inputSchema": {"type": "object", "properties": {"x": {"type": "integer"}}, "additionalProperties": False},
}


class ExtraBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__(image_size=16, max_steps=40)
        self.calls = []

    def extra_tools(self):
        return [PING]

    def call_extra(self, name, args, *, running):
        self.calls.append((name, dict(args), running))
        if args.get("x") == -1:
            raise McpToolError("design_rejected", "Rejected")
        return ExtraResult(data={"echo": args.get("x"), "running": running}, images=[np.zeros((8, 8, 3), np.uint8)], log={"secret": 1})


def test_extra_tool_is_listed_forwarded_and_logged(tmp_path):
    backend = ExtraBackend()
    session = McpSession(backend, output_root=tmp_path)
    assert "ping_body" in [tool["name"] for tool in session.tools()]

    result = session.call_tool("ping_body", {"x": 3})
    assert result["structuredContent"] == {"ok": True, "echo": 3, "running": False}
    assert [block["type"] for block in result["content"]] == ["text", "image"]
    # Log-only fields must not appear in the result
    assert "secret" not in json.dumps(result["structuredContent"])

    session.call_tool("reset_task", {})
    assert session.call_tool("ping_body", {"x": 4})["structuredContent"]["running"] is True

    log = [json.loads(line) for line in (tmp_path / "extra_tools.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [entry["arguments"]["x"] for entry in log] == [3, 4]
    assert log[0]["secret"] == 1 and log[1]["last_run"] == "r1"


def test_extra_tool_errors_come_back_as_tool_errors(tmp_path):
    session = McpSession(ExtraBackend(), output_root=tmp_path)
    result = session.call_tool("ping_body", {"x": -1})
    assert result["isError"] is True
    assert result["structuredContent"]["error"] == "design_rejected"


def test_backends_without_extras_still_reject_unknown_tools(tmp_path):
    session = McpSession(FakeBackend(image_size=16, max_steps=40), output_root=tmp_path)
    assert session.call_tool("ping_body", {})["structuredContent"]["error"] == "unknown_tool"


class AuditBackend(FakeBackend):
    def episode_audit(self):
        return {"max_penetration_m": 0.018, "ok": False}


def test_episode_audit_goes_to_summary_not_to_agent(tmp_path):
    session = McpSession(AuditBackend(image_size=16, max_steps=40), output_root=tmp_path)
    session.call_tool("reset_task", {})
    result = session.call_tool("stop_episode", {})
    assert "physics_audit" not in json.dumps(result)
    summary = json.loads(open(result["structuredContent"]["artifacts"]["summary"], encoding="utf-8").read())
    assert summary["physics_audit"] == {"max_penetration_m": 0.018, "ok": False}
