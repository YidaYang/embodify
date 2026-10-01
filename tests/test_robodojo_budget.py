"""Operator budgets propagate through metadata, worker and environment, without GPU."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodify_mcp import mcp
from embodify_mcp.backend.robodojo import RoboDojoBackend
from embodify_mcp.robodojo.config import Settings
from embodify_mcp.robodojo.episode import EpisodeBackend, info
from embodify_mcp.robodojo.isaac_driver import IsaacDriver
from tests.test_robodojo_backend import Driver
from tests.test_remote_backend import service


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "10000", None])
def test_operator_rejects_invalid_budget(value):
    if value is not None:  # None selects each task's native limit
        with pytest.raises(ValueError, match="positive integer"):
            Settings(source="s", python="p", output="o", episode_budget=value)
    with pytest.raises(ValueError): info(value)


def test_default_preserves_native_budget_and_applies_override_to_environment():
    settings = Settings(source="s", python="p", output="o")
    assert settings.task == "stack_blocks" and settings.episode_budget is None
    assert IsaacDriver(settings).budget == 550
    driver = IsaacDriver(Settings(source="s", python="p", output="o", episode_budget=10000))
    assert driver.budget == 10000  # worker handshake happens BEFORE reset
    driver.env = SimpleNamespace(step_lim=550, take_action_cnt=[0])
    driver._apply_episode_budget()
    assert driver.env.step_lim == driver.budget == 10000
    assert driver.native_budget == 550 and driver.env.take_action_cnt == [0]
    assert "isaacsim" not in sys.modules and "isaaclab" not in sys.modules


def test_actions_cross_old_limit_and_stop_at_new_limit(tmp_path):
    class Limited(Driver):
        count = 0
        def step(self, action):
            result = super().step(action)
            self.count += 1
            self.ended = self.count >= self.budget
            return result
    driver = Limited(budget=10000)
    b = EpisodeBackend(driver)
    s = mcp.McpSession(b, output_root=tmp_path)
    try:
        assert s.info.episode_budget == 10000
        assert not s.call_tool("reset_task").get("isError")
        b._step = driver.count = 549
        args = {"arms": {"left": {"delta_xyz": [.032, 0, 0]},
                         "right": {"delta_xyz": [0, 0, .032]}}}
        r = s.call_tool("control_arms", args)
        assert not r.get("isError"), r
        assert 550 < r["structuredContent"]["step"] < 560 and not driver.ended
        b._step = driver.count = 9999
        r = s.call_tool("control_arms", args)
        assert r["structuredContent"]["step"] == 10000 and driver.ended
        assert r["structuredContent"]["steps_left"] == 0
        count = len(driver.actions)
        assert s.call_tool("control_arms", args)["isError"]
        assert len(driver.actions) == count
    finally: s.close()


def test_budget_survives_two_remote_layers(tmp_path):
    cfg = tmp_path/"config.json"
    cfg.write_text(json.dumps({"source":"unused", "python":sys.executable,
                               "output":str(tmp_path), "episode_budget":10000}))
    code = ('import sys;sys.path.insert(0,' + repr(str(Path(__file__).parent)) + ');'
            'from test_robodojo_backend import Driver;'
            'from embodify_mcp.robodojo.episode import EpisodeBackend;'
            'from embodify_mcp.backend.serve import serve_connection;'
            'from embodify_mcp.backend.transport import LineTransport;'
            'd=Driver(budget=10000);'
            'serve_connection(EpisodeBackend(d),LineTransport(sys.stdin.buffer,sys.stdout.buffer,lambda:None))')
    backend = RoboDojoBackend(cfg, worker_command=[sys.executable,"-u","-c",code])
    assert backend.info.episode_budget == 10000 and backend.worker is None
    with service(backend) as (remote, _):
        s = mcp.McpSession(remote, output_root=tmp_path/"logs")
        try:
            assert remote.info.episode_budget == 10000
            assert not s.call_tool("reset_task").get("isError")
            assert backend.worker.info.episode_budget == 10000
            r = s.call_tool("get_session_info")["structuredContent"]
            assert r["max_steps"] == r["steps_left"] == 10000
            r = s.call_tool("control_arms", {"arms":{"left":{"delta_xyz":[0,0,.032]}}})
            assert not r.get("isError"), r
            assert r["structuredContent"]["steps_left"] == 10000 - r["structuredContent"]["action"]["executed_steps"]
            assert not s.call_tool("stop_episode").get("isError")
        finally: s.close()
    assert backend.worker is None


def test_frontend_refuses_stale_remote_budget_before_an_episode(monkeypatch, tmp_path):
    b = EpisodeBackend(Driver())
    monkeypatch.setattr(mcp, "build_backend", lambda args:b)
    args = mcp.build_parser().parse_args(["--expected-episode-budget","10000","--output-root",str(tmp_path)])
    with pytest.raises(ValueError, match="550.*10000"):
        mcp.build_session(args)
    assert b.driver.closed and not b.driver.actions and not list(tmp_path.iterdir())


def test_frontend_accepts_expected_budget_without_starting_simulation(monkeypatch, tmp_path):
    cfg = tmp_path/"config.json"
    cfg.write_text(json.dumps({"source":"unused", "python":sys.executable,
                               "output":str(tmp_path), "episode_budget":10000}))
    backend = RoboDojoBackend(cfg)
    monkeypatch.setattr(mcp,"build_backend",lambda args:backend)
    args = mcp.build_parser().parse_args(["--expected-episode-budget","10000"])
    s = mcp.build_session(args)
    try:
        assert s.info.episode_budget == 10000 and s.status == "idle" and backend.worker is None
    finally:s.close()
