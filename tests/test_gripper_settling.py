"""Delayed, obstructed and budget-limited gripper responses (FakeSim is instantaneous)."""

from dataclasses import replace

import pytest

from embodify_mcp.backend.fake import FakeBackend
from embodify_mcp.mcp import McpSession
from embodify_mcp.sim.fake import FakeSim


class ProfileGripperSim(FakeSim):
    def __init__(self, widths, max_steps=100):
        super().__init__(image_size=16, max_steps=max_steps)
        self.widths = widths

    def observe(self):
        obs = super().observe()
        return replace(obs, gripper_width=self.widths[min(obs.step_index, len(self.widths) - 1)])


def session_for(tmp_path, widths, *, per_call=30, budget=100, command="close"):
    session = McpSession(
        FakeBackend(ProfileGripperSim(widths, max_steps=budget)),
        output_root=tmp_path,
        max_steps_per_call=per_call,
    )
    session.call_tool("reset_task")
    session.backend.gripper_command = command
    return session


@pytest.mark.parametrize("state", ["open", "close"])
def test_initial_quiet_steps_are_not_completion(tmp_path, state):
    # First three readings from the failed release at LIBERO Long task 2, step 439.
    # Subsequent readings from replaying that exact state and continuing open (2026-09-26).
    # Closing mirrors the response to exercise the same startup gate in both directions.
    widths = [0.0141334, 0.0141583, 0.0144082, 0.0172796, 0.0231017, 0.0308240,
              0.0394383, 0.0485852, 0.0582218, 0.0665744, 0.0715837, 0.0744097,
              0.0759923, 0.0768776, 0.0773726, 0.0776494, 0.0778040]
    if state == "close":
        widths = [0.08 - w for w in widths]
    session = session_for(tmp_path, widths, command="close" if state == "open" else "open")
    result = session.call_tool("set_gripper", {"state": state})["structuredContent"]
    assert result["action"]["stop_reason"] == "settled"
    assert result["action"]["opening_after_m"] == pytest.approx(widths[-1])


def test_slow_continuous_motion_does_not_count_as_settled(tmp_path):
    session = session_for(tmp_path, [0.014 + 0.0003 * n for n in range(40)])
    result = session.call_tool("set_gripper", {"state": "open"})["structuredContent"]
    assert result["action"]["stop_reason"] == "step_cap"
    assert result["action"]["executed_steps"] == 30


def test_obstructed_gripper_waits_through_startup_but_still_stops(tmp_path):
    session = session_for(tmp_path, [0.014] * 40)
    result = session.call_tool("set_gripper", {"state": "open"})["structuredContent"]
    assert result["action"]["stop_reason"] == "settled"
    assert 8 < result["action"]["executed_steps"] < 30
    assert result["state"]["gripper"]["opening_m"] == 0.014


@pytest.mark.parametrize("per_call", [1, 2, 3])
def test_step_capped_command_can_continue_and_then_become_noop(tmp_path, per_call):
    session = session_for(tmp_path, [0.014, 0.0141, 0.0142, 0.03, 0.05, 0.07, 0.08], per_call=per_call)
    results = [session.call_tool("set_gripper", {"state": "open"})["structuredContent"] for _ in range(15)]
    assert results[0]["action"]["stop_reason"] == "step_cap"
    assert results[1]["action"]["executed_steps"] > 0
    assert any(r["action"]["stop_reason"] == "settled" for r in results)
    assert results[-1]["action"]["stop_reason"] == "no_op"
    assert results[-1]["state"]["gripper"]["opening_m"] == 0.08


def test_startup_is_bounded_by_episode_budget_and_retry_is_not_noop(tmp_path):
    session = session_for(tmp_path, [0.014] * 10, budget=2)
    first = session.call_tool("set_gripper", {"state": "open"})["structuredContent"]
    assert first["action"]["stop_reason"] == "budget"
    assert first["step"] == 2
    retry = session.call_tool("set_gripper", {"state": "open"})
    assert retry["isError"]
    assert retry["structuredContent"]["error"] == "step_limit"


def test_pending_gripper_state_is_cleared_on_reset(tmp_path):
    session = session_for(tmp_path, [0.014] * 40, per_call=1)
    session.call_tool("set_gripper", {"state": "open"})
    session.call_tool("stop_episode")
    session.call_tool("reset_task")
    result = session.call_tool("set_gripper", {"state": "open"})["structuredContent"]
    assert result["action"]["stop_reason"] == "no_op"


def test_reversing_command_restarts_startup_protection(tmp_path):
    session = session_for(tmp_path, [0.014] * 40)
    opened = session.call_tool("set_gripper", {"state": "open"})["structuredContent"]
    assert opened["action"]["stop_reason"] == "settled"
    session.max_steps_per_call = 2
    reversed_action = session.call_tool("set_gripper", {"state": "close"})["structuredContent"]
    assert reversed_action["action"]["stop_reason"] == "step_cap"
    assert reversed_action["action"]["executed_steps"] == 2


def test_pending_gripper_retry_is_isolated_per_arm(tmp_path, monkeypatch):
    from embodify_mcp.backend.base import LIMIT
    from embodify_mcp.backend.fake import FakeTwoArmBackend

    backend = FakeTwoArmBackend(image_size=16)
    session = McpSession(backend, output_root=tmp_path)
    session.call_tool("reset_task")
    original = backend.set_gripper
    limited_once = False

    def limit_first_call(arm, close, **kwargs):
        nonlocal limited_once
        report = original(arm, close, **kwargs)
        if not limited_once:
            limited_once = True
            return replace(report, stop_reason=LIMIT)
        return report

    monkeypatch.setattr(backend, "set_gripper", limit_first_call)
    first = session.call_tool("set_gripper", {"arm": "right", "state": "close"})["structuredContent"]
    assert first["action"]["stop_reason"] == "step_cap"
    left = session.call_tool("set_gripper", {"arm": "left", "state": "open"})["structuredContent"]
    assert left["action"]["stop_reason"] == "no_op"
    retry = session.call_tool("set_gripper", {"arm": "right", "state": "close"})["structuredContent"]
    assert retry["action"]["stop_reason"] == "settled"
    assert retry["action"]["executed_steps"] > 0
