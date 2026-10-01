"""Backend contract tests: every backend connected to MCP must pass them.

To add a backend (a simulator bridge, SO-101, a remote backend), add its factory to `BACKENDS`.
Two layers:
- the `Backend` interface itself (steps, callbacks and stop reasons of move_to / set_gripper, ...);
- a full episode through `McpSession`, checking the tool list and every result: the number of
  images, whether the arm parameter is present, and that no word the backend declared secret leaks.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from embodify_mcp.backend.base import LIMIT, Backend, Snapshot
from embodify_mcp.backend.fake import FakeBackend, FakeTwoArmBackend
from embodify_mcp.mcp import McpSession

#: Words no backend may leak to the Agent. Each backend adds its own in info.forbidden_terms.
COMMON_FORBIDDEN = ("success", "privileged")

#: Stop reasons the MCP session reports itself. Backend reasons must not reuse them, or the Agent could not tell who stopped.
SESSION_REASONS = {"step_cap", "budget", "no_op", LIMIT}


def _libero_wanted() -> bool:
    """Real LIBERO contract tests must be enabled explicitly: EMBODIFY_LIBERO_CONTRACT=1.

    They need LIBERO, robosuite and MuJoCo installed and take about a minute. Run them with:
        EMBODIFY_LIBERO_CONTRACT=1 python -m pytest tests/test_backend_contract.py -k libero
    """
    return (
        os.environ.get("EMBODIFY_LIBERO_CONTRACT") == "1"
        and importlib.util.find_spec("libero") is not None
        and importlib.util.find_spec("robosuite") is not None
    )


_LIBERO = []


def _libero():
    """One LiberoBackend for the whole test process, as in an MCP server process."""
    if not _LIBERO:
        from embodify_mcp.backend.libero import LiberoBackend

        _LIBERO.append(LiberoBackend(image_size=64, max_steps=200))
    return _LIBERO[0]


def _remote(name):
    from embodify_mcp.backend.remote import RemoteBackend
    return RemoteBackend(command=[sys.executable, "-m", "embodify_mcp.backend.serve", "--backend", name,
                                  "--image-size", "16", "--max-steps", "80"],
                         cwd=str(Path(__file__).resolve().parents[1]))


BACKENDS = [
    pytest.param("robodojo_mock", id="robodojo_mock"),
    pytest.param(lambda: _remote("fake"), id="remote_fake"),
    pytest.param(lambda: _remote("fake-two-arm"), id="remote_two_arm"),
    pytest.param(lambda: FakeBackend(image_size=16, max_steps=80), id="fake"),
    pytest.param(lambda: FakeTwoArmBackend(image_size=16, max_steps=80), id="fake_two_arm"),
    pytest.param(
        _libero,
        id="libero",
        marks=pytest.mark.skipif(not _libero_wanted(), reason="needs EMBODIFY_LIBERO_CONTRACT=1 and LIBERO installed"),
    ),
]


@pytest.fixture(params=BACKENDS)
def backend(request, tmp_path) -> Backend:
    if request.param == "robodojo_mock":
        from tests.test_robodojo_backend import make_supervisor
        made = make_supervisor(tmp_path)
    else:
        made = request.param()
    yield made
    try:
        made.close()
    except Exception:
        pass


def _start(backend: Backend) -> Snapshot:
    task, variant = backend.catalogue.resolve({}, locked=False)
    backend.catalogue.commit(task, variant)
    return backend.reset(task, variant)


# -- the interface itself ----------------------------------------------------------


def test_info_is_self_consistent(backend):
    info = backend.info
    assert info.name and info.title and info.server_name and info.frame
    assert info.arms, "at least one arm"
    assert len(set(info.arm_names)) == len(info.arm_names), "arm names must be unique"
    names = [camera.name for camera in info.cameras]
    assert names and len(set(names)) == len(names), "at least one camera, with unique names"
    assert int(info.episode_budget) > 0
    declared = {name for name, _ in info.move_stop_reasons} | {name for name, _ in info.gripper_stop_reasons}
    assert not declared & SESSION_REASONS, "backend stop reasons must not reuse session reasons"
    for arm in info.arms:
        assert set(arm.rotation_axes) <= {"x", "y", "z"}


def test_reset_gives_every_arm_and_every_camera(backend):
    snap = _start(backend)
    info = backend.info
    assert snap.step == 0
    assert set(snap.arms) == set(info.arm_names)
    for camera in info.cameras:
        image = snap.images[camera.name]
        assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8
    for arm in snap.arms.values():
        assert np.asarray(arm.pos).shape == (3,)
        assert np.asarray(arm.quat).shape == (4,)
        assert abs(float(np.linalg.norm(arm.quat)) - 1.0) < 1e-6
        assert arm.gripper_command == "open", "the gripper command should be open after reset"
        if info.gripper_opening_measured:
            assert arm.gripper_opening >= 0.0
        else:
            assert arm.gripper_opening is None


def test_observe_does_not_spend_budget(backend):
    first = _start(backend)
    again = backend.observe()
    assert again.step == first.step


def test_move_reports_every_step_it_takes(backend):
    snap = _start(backend)
    arm = backend.info.arm_names[0]
    start = snap.arms[arm]
    seen = []
    report = backend.move_to(
        arm, np.asarray(start.pos) + np.array([0.0, 0.0, 0.03]), start.quat, max_steps=20, on_step=seen.append
    )
    assert 1 <= report.executed_steps <= 20
    assert len(seen) == report.executed_steps, "one callback per step (live frame recording relies on it)"
    assert [s.step for s in seen] == list(range(snap.step + 1, snap.step + 1 + report.executed_steps))
    assert report.snapshot.step == snap.step + report.executed_steps
    declared = {name for name, _ in backend.info.move_stop_reasons}
    assert report.stop_reason in declared | {LIMIT}
    assert report.remaining_distance_m >= 0.0
    moved = float(report.snapshot.arms[arm].pos[2] - start.pos[2])
    assert moved > 0.005, "a 3 cm move up must move at least a little"


def test_move_never_exceeds_its_allowance(backend):
    snap = _start(backend)
    arm = backend.info.arm_names[0]
    start = snap.arms[arm]
    report = backend.move_to(arm, np.asarray(start.pos) + np.array([0.0, 0.0, 0.5]), start.quat, max_steps=1)
    assert report.executed_steps == 1
    assert report.stop_reason == LIMIT, "using the whole allowance without arriving reports LIMIT, which the session turns into step_cap / budget"


def test_gripper_latch_shows_up_in_the_snapshot(backend):
    _start(backend)
    arm = backend.info.arm_names[0]
    report = backend.set_gripper(arm, True, max_steps=10)
    assert 1 <= report.executed_steps <= 10
    assert report.snapshot.arms[arm].gripper_command == "close"
    declared = {name for name, _ in backend.info.gripper_stop_reasons}
    assert report.stop_reason in declared | {LIMIT}
    # After closing, a move must keep the close command (it rides along on every step)
    start = report.snapshot.arms[arm]
    moved = backend.move_to(arm, np.asarray(start.pos) + np.array([0.02, 0.0, 0.0]), start.quat, max_steps=5)
    assert moved.snapshot.arms[arm].gripper_command == "close"


def test_success_is_a_plain_bool_when_supported(backend):
    _start(backend)
    if backend.info.has_success:
        assert backend.success() in (True, False)


def test_episode_can_be_ended_and_started_again(backend):
    _start(backend)
    backend.end_episode()
    again = _start(backend)
    assert again.step == 0


def test_catalogue_lists_what_it_can_start(backend):
    top = backend.catalogue.listing({})
    assert top["ok"] is True
    task, variant = backend.catalogue.resolve({}, locked=False)
    backend.catalogue.commit(task, variant)
    payload = backend.catalogue.task_payload()
    assert payload["instruction"] == task.instruction


# -- a full episode through the MCP session -------------------------------------------


def _episode(session: McpSession):
    """A standard episode; returns the tool list and the result of every call."""
    multi = len(session.info.arms) > 1
    arm = {"arm": session.info.arm_names[0]} if multi else {}
    results = [
        session.call_tool("get_session_info"),
        session.call_tool("list_tasks"),
        session.call_tool("reset_task"),
        session.call_tool("observe"),
        session.call_tool("move_relative", dict(arm, delta_xyz=[0.0, 0.0, 0.02])),
        session.call_tool("set_gripper", dict(arm, state="close")),
        session.call_tool("get_session_info"),
        session.call_tool("stop_episode"),
    ]
    return session.tools(), results


def test_a_full_episode_leaks_nothing_the_backend_declared_secret(backend, tmp_path):
    session = McpSession(backend, output_root=tmp_path / "mcp", max_steps_per_call=10)
    tools, results = _episode(session)
    assert all(not r.get("isError") for r in results), [r for r in results if r.get("isError")]
    blob = json.dumps([tools] + [r["structuredContent"] for r in results], ensure_ascii=False).lower()
    for term in COMMON_FORBIDDEN + tuple(backend.info.forbidden_terms):
        assert term.lower() not in blob, f"{term!r} leaked into what the Agent can see"


def test_every_picture_returning_call_carries_one_image_per_camera(backend, tmp_path):
    session = McpSession(backend, output_root=tmp_path / "mcp", max_steps_per_call=10)
    _, results = _episode(session)
    n = len(backend.info.cameras)
    for name, result in zip(["info", "list", "reset", "observe", "move", "gripper"], results):
        images = [c for c in result["content"] if c["type"] == "image"]
        expected = n if name in ("reset", "observe", "move", "gripper") else 0
        assert len(images) == expected, name
    described = [d["camera"] for d in results[2]["structuredContent"]["images"]]
    assert described == [c.name for c in backend.info.cameras]


def test_arm_parameter_appears_exactly_when_there_is_a_choice(backend, tmp_path):
    session = McpSession(backend, output_root=tmp_path / "mcp")
    by_name = {tool["name"]: tool["inputSchema"] for tool in session.tools()}
    multi = len(backend.info.arms) > 1
    for tool in ("move_relative", "set_gripper"):
        assert ("arm" in by_name[tool]["properties"]) is multi, tool
        assert ("arm" in by_name[tool]["required"]) is multi, tool


def test_frames_are_archived_for_every_camera(backend, tmp_path):
    session = McpSession(backend, output_root=tmp_path / "mcp", max_steps_per_call=10)
    session.call_tool("reset_task")
    written = {p.stem for p in (session.journal.dir / "images").iterdir()}
    for camera in backend.info.cameras:
        assert f"reset_{camera.name}" in written, camera.name
    scene = [e for e in session.journal.read_events() if e["kind"] == "scene"][0]
    assert [c["name"] for c in scene["cameras"]] == [c.name for c in backend.info.cameras]
    assert scene["arms"] == list(backend.info.arm_names)
