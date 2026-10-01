"""Multi-arm, multi-camera backends through the MCP session (RoboDojo, with two arms and three cameras, relies on this path).

Uses `FakeTwoArmBackend`: arms left / right, three cameras, and a right arm that controls only rotation about z.
"""

import json

import numpy as np

from embodify_mcp.backend.fake import FakeTwoArmBackend
from embodify_mcp.mcp import McpSession


def session_for(tmp_path, **kwargs):
    return McpSession(FakeTwoArmBackend(image_size=16, max_steps=80), output_root=tmp_path / "mcp", **kwargs)


def test_state_is_split_by_arm(tmp_path):
    session = session_for(tmp_path)
    state = session.call_tool("reset_task")["structuredContent"]["state"]
    assert set(state) == {"arms"}
    assert set(state["arms"]) == {"left", "right"}
    for arm in state["arms"].values():
        assert len(arm["eef_pos_m"]) == 3
        assert arm["gripper"] == {"command": "open", "opening_m": 0.08}


def test_moving_one_arm_leaves_the_other_alone(tmp_path):
    session = session_for(tmp_path, max_steps_per_call=10)
    before = session.call_tool("reset_task")["structuredContent"]["state"]["arms"]
    moved = session.call_tool("move_relative", {"arm": "left", "delta_xyz": [0.0, -0.1, 0.0]})["structuredContent"]
    after = moved["state"]["arms"]
    assert moved["action"]["arm"] == "left"
    assert moved["action"]["stop_reason"] == "reached"
    assert abs(after["left"]["eef_pos_m"][1] - (before["left"]["eef_pos_m"][1] - 0.1)) < 0.01
    assert after["right"]["eef_pos_m"] == before["right"]["eef_pos_m"]
    assert abs(moved["action"]["achieved_delta_xyz"][1] + 0.1) < 0.01


def test_arm_is_required_and_checked(tmp_path):
    session = session_for(tmp_path)
    session.call_tool("reset_task")
    missing = session.call_tool("move_relative", {"delta_xyz": [0.0, 0.0, 0.01]})["structuredContent"]
    assert missing["error"] == "invalid_input" and "left" in missing["message"] and "right" in missing["message"]
    wrong = session.call_tool("set_gripper", {"arm": "middle", "state": "close"})["structuredContent"]
    assert wrong["error"] == "invalid_input"


def test_each_gripper_has_its_own_latch(tmp_path):
    session = session_for(tmp_path)
    session.call_tool("reset_task")
    closed = session.call_tool("set_gripper", {"arm": "right", "state": "close"})["structuredContent"]
    assert closed["action"]["arm"] == "right"
    assert closed["state"]["arms"]["right"]["gripper"]["command"] == "close"
    assert closed["state"]["arms"]["left"]["gripper"]["command"] == "open"
    again = session.call_tool("set_gripper", {"arm": "right", "state": "close"})["structuredContent"]
    assert again["action"]["stop_reason"] == "no_op"
    other = session.call_tool("set_gripper", {"arm": "left", "state": "close"})["structuredContent"]
    assert other["action"]["stop_reason"] == "settled", "the left gripper is still open, so closing it is not a no_op"


def test_three_cameras_come_back_in_declared_order(tmp_path):
    session = session_for(tmp_path)
    reset = session.call_tool("reset_task")
    assert [d["camera"] for d in reset["structuredContent"]["images"]] == ["cam_head", "cam_left_wrist", "cam_right_wrist"]
    assert len([c for c in reset["content"] if c["type"] == "image"]) == 3


def test_descriptions_speak_of_arms_and_limited_rotation(tmp_path):
    session = session_for(tmp_path)
    by_name = {tool["name"]: tool for tool in session.tools()}
    move = by_name["move_relative"]
    assert move["inputSchema"]["properties"]["arm"]["enum"] == ["left", "right"]
    assert "right can only rotate" in move["description"], "partial orientation control must be explained"
    for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        assert camera in by_name["reset_task"]["description"]
    assert "arms" in by_name["observe"]["description"]


def test_right_arm_rotation_is_only_honoured_about_z(tmp_path):
    session = session_for(tmp_path, max_steps_per_call=10)
    session.call_tool("reset_task")
    moved = session.call_tool(
        "move_relative", {"arm": "right", "delta_xyz": [0.0, 0.0, 0.0], "delta_rpy": [0.3, 0.0, 0.2]}
    )["structuredContent"]
    rpy = moved["state"]["arms"]["right"]["eef_rpy_rad"]
    assert abs(rpy["roll_x"]) < 1e-6, "the right arm cannot rotate about x"
    assert abs(rpy["yaw_z"] - 0.2) < 0.03


def test_handover_verdict_is_recorded_but_never_shown(tmp_path):
    session = session_for(tmp_path, max_steps_per_call=20)
    session.call_tool("reset_task")
    responses = [
        session.call_tool("move_relative", {"arm": "left", "delta_xyz": [0.0, -0.48, 0.0]}),
        session.call_tool("set_gripper", {"arm": "left", "state": "close"}),
        session.call_tool("set_gripper", {"arm": "right", "state": "close"}),
    ]
    stopped = session.call_tool("stop_episode")
    summary = json.loads(open(stopped["structuredContent"]["artifacts"]["summary"], encoding="utf-8").read())
    assert summary["success"] is True
    assert summary["backend"] == "fake_two_arm"
    blob = json.dumps([r["structuredContent"] for r in responses + [stopped]], ensure_ascii=False)
    assert "success" not in blob


def test_journal_move_record_names_the_arm(tmp_path):
    session = session_for(tmp_path)
    session.call_tool("reset_task")
    session.call_tool("move_relative", {"arm": "right", "delta_xyz": [0.01, 0.0, 0.0]})
    moves = [e for e in session.journal.read_events() if e["kind"] == "mcp_move"]
    assert moves[0]["arm"] == "right"
    observations = [e for e in session.journal.read_events() if e["kind"] == "observation"]
    assert set(observations[-1]["state"]["arms"]) == {"left", "right"}


def test_single_arm_output_shape_is_unchanged():
    """Single-arm state stays flat, without an arms wrapper, so a LIBERO Agent sees exactly what it did before."""
    from embodify_mcp.backend.fake import FakeBackend

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        session = McpSession(FakeBackend(image_size=16), output_root=Path(tmp) / "mcp")
        state = session.call_tool("reset_task")["structuredContent"]["state"]
        session.call_tool("stop_episode")
    assert set(state) == {"eef_pos_m", "eef_rpy_rad", "gripper"}
    assert np.isfinite(state["eef_pos_m"]).all()
