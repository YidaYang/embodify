import io
import json
import textwrap
from pathlib import Path

import numpy as np

from embodify_mcp.backend.fake import FakeBackend
from embodify_mcp.mcp import McpSession, rpc_response
from embodify_mcp.sim.fake import FakeSim, fake_tasks
from embodify_mcp.imaging import decode_png_size


def make_session(tmp_path: Path, *, image_size: int = 32) -> McpSession:
    return McpSession(
        FakeBackend(image_size=image_size, max_steps=80),
        output_root=tmp_path / "mcp",
        max_steps_per_call=3,
    )


#: A target that is never reached: FakeSim moves at most 0.05 m per step, so moving toward it always
#: runs at full speed, never converges early and never stops at the table. Budget-burning tests use it.
FAR_AWAY = [10.0, 0.0, 0.0]


def image_blocks(result):
    return [item for item in result["content"] if item["type"] == "image"]


def test_tools_list_matches_the_declared_set(tmp_path):
    """Regression guard for the tool list: changes must be deliberate, and the order matters (schemas are checked by index).

    There is **no** cap on the number of tools; add tools as needed, as long as each description
    explains itself.
    """
    session = make_session(tmp_path)
    tools = session.tools()
    assert [tool["name"] for tool in tools] == [
        "reset_task",
        "observe",
        "move_relative",
        "set_gripper",
        "stop_episode",
        "get_session_info",
        "list_tasks",
    ]
    # All three scene arguments of reset_task must be **optional**: without them the default scene is used.
    assert sorted(tools[0]["inputSchema"]["properties"]) == ["init_state_index", "suite", "task_index"]
    assert tools[0]["inputSchema"]["required"] == []
    assert tools[1]["inputSchema"]["properties"] == {}
    assert "eef_pos_m" in tools[0]["description"]
    assert "opening_m" in tools[0]["description"]
    assert "step" in tools[1]["description"]
    assert list(tools[2]["inputSchema"]["properties"]) == ["delta_xyz", "delta_rpy"]
    assert tools[2]["inputSchema"]["properties"]["delta_rpy"]["minItems"] == 3
    assert list(tools[3]["inputSchema"]["properties"]) == ["state"]


def test_reset_and_observe_return_two_independent_images_and_state(tmp_path):
    session = make_session(tmp_path, image_size=32)
    reset = session.call_tool("reset_task")
    assert reset["structuredContent"]["run"] == "r1"
    assert reset["structuredContent"]["task"]["suite"] == "fake_suite"
    assert len(image_blocks(reset)) == 2
    assert {item["camera"] for item in reset["structuredContent"]["images"]} == {"agentview", "robot0_eye_in_hand"}
    assert all(decode_png_size(__import__("base64").b64decode(item["data"])) == (32, 32) for item in image_blocks(reset))
    state = reset["structuredContent"]["state"]
    assert len(state["eef_pos_m"]) == 3
    assert set(state["eef_rpy_rad"]) == {"roll_x", "pitch_y", "yaw_z"}
    assert state["gripper"]["command"] == "open"
    assert "qpos" not in state["gripper"]

    observed = session.call_tool("observe")
    assert observed["structuredContent"]["step"] == 0
    assert len(image_blocks(observed)) == 2


def test_move_returns_new_state_and_keeps_gripper_latched(tmp_path):
    session = make_session(tmp_path)
    session.call_tool("reset_task")
    before = session.call_tool("observe")["structuredContent"]["state"]["eef_pos_m"]
    moved = session.call_tool(
        "move_relative",
        {"delta_xyz": [0.0, 0.0, -0.02], "delta_rpy": [0.0, 0.0, 0.2]},
    )
    after = moved["structuredContent"]["state"]["eef_pos_m"]
    assert moved["structuredContent"]["step"] == 3
    assert after[2] < before[2]
    assert abs(moved["structuredContent"]["state"]["eef_rpy_rad"]["yaw_z"]) > 0.05
    assert moved["structuredContent"]["state"]["gripper"]["command"] == "open"
    assert moved["structuredContent"]["action"]["delta_rpy"] == [0.0, 0.0, 0.2]
    assert "request_id" not in moved["structuredContent"]


def test_gripper_output_is_semantic_and_same_state_is_noop(tmp_path):
    session = make_session(tmp_path)
    session.max_steps_per_call = 30
    session.call_tool("reset_task")
    closed = session.call_tool("set_gripper", {"state": "close"})
    assert closed["structuredContent"]["action"]["stop_reason"] == "settled"
    assert closed["structuredContent"]["state"]["gripper"] == {"command": "close", "opening_m": 0.0}
    repeated = session.call_tool("set_gripper", {"state": "close"})
    assert repeated["structuredContent"]["step"] == closed["structuredContent"]["step"]
    assert repeated["structuredContent"]["action"]["stop_reason"] == "no_op"


def test_session_info_contains_task_instruction_and_state(tmp_path):
    session = make_session(tmp_path)
    idle = session.call_tool("get_session_info")["structuredContent"]
    assert idle["ok"] is True and idle["status"] == "idle"
    # The catalogue belongs to the benchmark, not to this run; list_tasks covers it.
    assert "available_scenes" not in idle and "suites" not in idle
    # default_scene must include suite: with the whole benchmark open, task_index alone is ambiguous.
    assert idle["default_scene"] == {"suite": "fake_suite", "task_index": 0, "init_state_index": 0}
    assert "state" not in idle
    session.call_tool("reset_task")
    info = session.call_tool("get_session_info")
    assert info["structuredContent"]["run"] == "r1"
    assert info["structuredContent"]["task"]["instruction"].startswith("pick up")
    assert "eef_pos_m" in info["structuredContent"]["state"]
    assert "success" not in json.dumps(info["structuredContent"])


def test_stop_writes_artifacts_and_rejects_later_observe(tmp_path):
    session = make_session(tmp_path)
    session.call_tool("reset_task")
    stopped = session.call_tool("stop_episode")
    assert stopped["structuredContent"]["status"] == "stopped"
    assert Path(stopped["structuredContent"]["artifacts"]["summary"]).exists()
    rejected = session.call_tool("observe")
    assert rejected["isError"] is True
    assert rejected["structuredContent"]["error"] == "not_running"


def test_stdio_rpc_transcript(tmp_path):
    session = make_session(tmp_path, image_size=16)
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "reset_task", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "observe", "arguments": {}}},
    ]
    from embodify_mcp.mcp import serve_stdio

    source = io.StringIO("\n".join(json.dumps(item) for item in requests) + "\n")
    sink = io.StringIO()
    serve_stdio(session, source, sink)
    responses = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert [item["id"] for item in responses] == [1, 2, 3, 4]
    assert len(responses[2]["result"]["content"]) == 3  # text + two images
    assert responses[3]["result"]["structuredContent"]["step"] == 0



def test_library_stdout_noise_does_not_corrupt_protocol_stream(tmp_path):
    """LIBERO's benchmark prints the task order to stdout, and MuJoCo may write to fd 1 from C.

    split_protocol_stdout must give the protocol stream its own fd, or the host's JSON-RPC parsing fails.
    """
    import subprocess
    import sys

    program = textwrap.dedent(
        """
        import json, os, sys
        sys.path.insert(0, %r)
        from embodify_mcp.mcp import split_protocol_stdout

        sink = split_protocol_stdout()
        print("[info] using task orders [0, 1, 2]")          # library noise from Python
        os.write(1, b"native library chatter\\n")            # C code writing to fd 1 directly
        sys.stderr.write("diagnostics stay on stderr\\n")
        sink.write(json.dumps({"jsonrpc": "2.0", "id": 1}) + "\\n")
        sink.flush()
        """
    ) % str(Path(__file__).resolve().parents[1])

    proc = subprocess.run(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout = proc.stdout.decode("utf-8")
    stderr = proc.stderr.decode("utf-8")
    assert proc.returncode == 0, stderr

    lines = [line for line in stdout.splitlines() if line.strip()]
    assert lines == [json.dumps({"jsonrpc": "2.0", "id": 1})]
    for line in lines:
        json.loads(line)  # every line of the protocol stream must be valid JSON
    assert "using task orders" in stderr
    assert "native library chatter" in stderr
    assert "diagnostics stay on stderr" in stderr


# -- step budget, gripper latch, gripper sign and controller constants ---------------


def exhausted_session(tmp_path, *, max_steps: int = 6, max_steps_per_call: int = 4):
    """A session whose step budget is about to run out."""
    session = McpSession(
        FakeBackend(image_size=16, max_steps=max_steps),
        output_root=tmp_path / "mcp",
        max_steps_per_call=max_steps_per_call,
    )
    session.call_tool("reset_task")
    return session


def drain(session):
    """Push toward an unreachable target until the step budget is gone."""
    while session.call_tool("get_session_info")["structuredContent"]["steps_left"] > 0:
        session.call_tool("move_relative", {"delta_xyz": FAR_AWAY})


def test_move_after_budget_is_gone_reports_step_limit_instead_of_silent_noop(tmp_path):
    """With the budget gone, move used to return ok=true without moving, leaving the Agent to retry forever."""
    session = exhausted_session(tmp_path)
    drain(session)

    refused = session.call_tool("move_relative", {"delta_xyz": FAR_AWAY})
    assert refused["isError"] is True
    assert refused["structuredContent"]["error"] == "step_limit"
    assert "stop_episode" in refused["structuredContent"]["message"]


def test_partial_budget_executes_what_is_left_and_reports_it(tmp_path):
    session = exhausted_session(tmp_path)  # budget 6, per-call cap 4
    first = session.call_tool("move_relative", {"delta_xyz": FAR_AWAY})
    assert first["structuredContent"]["action"]["executed_steps"] == 4
    assert first["structuredContent"]["action"]["stop_reason"] == "step_cap"
    assert first["structuredContent"]["steps_left"] == 2

    second = session.call_tool("move_relative", {"delta_xyz": FAR_AWAY})
    assert second["structuredContent"]["action"]["executed_steps"] == 2  # only 2 steps left
    assert second["structuredContent"]["action"]["stop_reason"] == "budget"
    assert second["structuredContent"]["steps_left"] == 0


def test_gripper_latch_does_not_flip_when_the_budget_is_gone(tmp_path):
    """The latch used to flip before executing 0 steps, reporting command=close with the gripper wide open."""
    session = exhausted_session(tmp_path)
    drain(session)
    opening_before = session.call_tool("get_session_info")["structuredContent"]["state"]["gripper"]["opening_m"]

    refused = session.call_tool("set_gripper", {"state": "close"})
    assert refused["isError"] is True
    assert refused["structuredContent"]["error"] == "step_limit"

    gripper = session.call_tool("get_session_info")["structuredContent"]["state"]["gripper"]
    assert gripper["command"] == "open", "the gripper did not move, so command must not become close"
    assert gripper["opening_m"] == opening_before


def test_repeating_the_current_gripper_state_still_works_without_budget(tmp_path):
    """Repeating the current state needs no action, so an empty budget must not cause an error."""
    session = exhausted_session(tmp_path)
    drain(session)
    repeated = session.call_tool("set_gripper", {"state": "open"})
    assert repeated.get("isError") is None
    assert repeated["structuredContent"]["action"]["executed_steps"] == 0
    assert repeated["structuredContent"]["action"]["stop_reason"] == "no_op"


class RecordingSim:
    """Records every action sent to the simulator, to check the sign of the gripper dimension."""

    def __init__(self, inner):
        self.inner = inner
        self.actions = []

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=float).copy())
        return self.inner.step(action)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_move_takes_the_gripper_sign_from_controller_params_not_a_hardcoded_plus_one(tmp_path):
    """move once hard-coded +1 = close; with the opposite sign the gripper would open during moves while set_gripper would not."""
    from embodify_mcp.primitives.params import ControllerParams

    sim = RecordingSim(FakeSim(image_size=16, max_steps=80))
    session = McpSession(FakeBackend(sim), output_root=tmp_path / "mcp", max_steps_per_call=2)
    session.backend.controller_params = ControllerParams(gripper_close_sign=-1.0)
    session.call_tool("reset_task")
    session.backend.gripper_command = "close"

    sim.actions.clear()
    session.call_tool("move_relative", {"delta_xyz": [0.0, 0.0, -0.01]})
    assert sim.actions, "move should actually step the simulation"
    for action in sim.actions:
        assert action[-1] == -1.0, "the close sign must come from ControllerParams"


def test_controller_constants_are_not_claimed_verified_by_default(tmp_path):
    """gripper_close_sign is unmeasured by default, so the server must not claim it is verified."""
    session = make_session(tmp_path)
    assert session.backend.controller_params.verified_against_facts is False


def test_run_journal_records_the_controller_constants_and_their_provenance(tmp_path):
    session = make_session(tmp_path)
    session.call_tool("reset_task")
    events = session.journal.read_events()
    recorded = [item for item in events if item["kind"] == "controller_params"]
    assert len(recorded) == 1
    assert recorded[0]["verified_against_facts"] is False
    assert recorded[0]["gripper_close_sign"] == 1.0
    assert recorded[0]["max_steps"] == 80


# -- convergent execution and stop reasons --------------------------------------


def convergence_session(tmp_path, *, max_steps: int = 200, max_steps_per_call: int = 30):
    session = McpSession(
        FakeBackend(image_size=16, max_steps=max_steps),
        output_root=tmp_path / "mcp",
        max_steps_per_call=max_steps_per_call,
    )
    session.call_tool("reset_task")
    return session


def test_move_stops_at_the_target_instead_of_burning_the_whole_cap(tmp_path):
    """The old implementation always ran a fixed number of steps and kept burning budget after arriving."""
    from embodify_mcp.backend.stepped import POSITION_TOLERANCE_M

    session = convergence_session(tmp_path)
    moved = session.call_tool("move_relative", {"delta_xyz": [0.0, 0.0, -0.02]})
    action = moved["structuredContent"]["action"]
    assert action["stop_reason"] == "reached"
    assert 0 < action["executed_steps"] < 30, "must stop once arrived instead of using the whole cap"
    assert action["remaining_distance_m"] <= POSITION_TOLERANCE_M


def test_move_reports_stalled_when_the_table_blocks_the_hand(tmp_path):
    """FakeSim keeps the end effector above the table, so pushing down is a real block.

    The old results could not tell "hit something" from "controller too weak".
    """
    session = convergence_session(tmp_path)
    session.call_tool("move_relative", {"delta_xyz": [0.0, 0.0, -0.5]})  # first reach the table
    blocked = session.call_tool("move_relative", {"delta_xyz": [0.0, 0.0, -0.5]})
    action = blocked["structuredContent"]["action"]
    assert action["stop_reason"] == "stalled"
    assert action["executed_steps"] < 30, "a stall must stop early instead of using the whole cap"
    assert action["remaining_distance_m"] > 0.01, "stalled implies the target was not reached"
    assert abs(action["achieved_delta_xyz"][2]) < 0.01


def test_move_reports_step_cap_while_still_travelling(tmp_path):
    session = convergence_session(tmp_path, max_steps_per_call=5)
    moved = session.call_tool("move_relative", {"delta_xyz": FAR_AWAY})
    action = moved["structuredContent"]["action"]
    assert action["stop_reason"] == "step_cap"
    assert action["executed_steps"] == 5
    assert action["achieved_delta_xyz"][0] > 0.2, "5 full-speed steps cover at least 0.25 m"


def test_achieved_delta_is_reported_so_the_agent_need_not_diff_states(tmp_path):
    session = convergence_session(tmp_path)
    before = session.call_tool("observe")["structuredContent"]["state"]["eef_pos_m"]
    moved = session.call_tool("move_relative", {"delta_xyz": [0.0, 0.0, -0.03]})
    after = moved["structuredContent"]["state"]["eef_pos_m"]
    achieved = moved["structuredContent"]["action"]["achieved_delta_xyz"]
    for axis in range(3):
        assert abs(achieved[axis] - (after[axis] - before[axis])) < 1e-9


def test_move_journal_records_per_step_distances_for_calibration(tmp_path):
    """The per-step record lets measurements replace the estimated step cap and stall threshold."""
    session = convergence_session(tmp_path)
    session.call_tool("move_relative", {"delta_xyz": [0.0, 0.0, -0.02]})
    moves = [item for item in session.journal.read_events() if item["kind"] == "mcp_move"]
    assert len(moves) == 1
    record = moves[0]
    assert len(record["step_distances_m"]) == record["executed_steps"]
    assert len(record["step_errors_m"]) == record["executed_steps"]
    assert record["step_errors_m"] == sorted(record["step_errors_m"], reverse=True)
    assert record["stop_reason"] == "reached"


def test_gripper_stops_once_the_opening_settles(tmp_path):
    session = convergence_session(tmp_path)
    closed = session.call_tool("set_gripper", {"state": "close"})
    action = closed["structuredContent"]["action"]
    assert action["stop_reason"] == "settled"
    assert 0 < action["executed_steps"] < 30
    assert action["opening_before_m"] > action["opening_after_m"]


def test_gripper_budget_exhaustion_is_named_budget_not_settled(tmp_path):
    session = exhausted_session(tmp_path, max_steps=2, max_steps_per_call=30)
    closed = session.call_tool("set_gripper", {"state": "close"})
    assert closed["structuredContent"]["action"]["stop_reason"] == "budget"
    assert closed["structuredContent"]["steps_left"] == 0


def test_every_stop_reason_is_explained_in_the_tool_description(tmp_path):
    """The Agent should not have to guess what these values mean, so each one must be in the description."""
    session = make_session(tmp_path)
    by_name = {tool["name"]: tool["description"] for tool in session.tools()}
    for reason in ("reached", "stalled", "step_cap", "budget"):
        assert reason in by_name["move_relative"], reason
    for reason in ("settled", "step_cap", "budget", "no_op"):
        assert reason in by_name["set_gripper"], reason
    assert "stop_reason" in by_name["move_relative"]
    assert "achieved_delta_xyz" in by_name["move_relative"]
    assert "remaining_distance_m" in by_name["move_relative"]


def test_default_episode_budget_is_the_full_libero_horizon():
    """LIBERO's OffScreenRenderEnv defaults to a horizon of 1000, and the budget should not be lower."""
    from embodify_mcp.mcp import build_parser

    defaults = build_parser().parse_args([])
    assert defaults.max_steps == 1000
    assert defaults.max_steps_per_call == 30


# -- archived frames and scene selection by the Agent --------------------------------


def multi_scene_session(tmp_path, *, locked: bool = False, default_index: int = 0):
    """A session with 4 selectable scenes (built from FakeSim's fake_tasks)."""
    tasks = fake_tasks(4)
    backend = FakeBackend(
        FakeSim(image_size=16, max_steps=80),
        tasks=tasks,
        task_source=lambda suite: fake_tasks(4, suite=suite),
        default_task_index=default_index,
    )
    return McpSession(backend, output_root=tmp_path / "mcp", max_steps_per_call=3, task_locked=locked)


def test_session_writes_frames_with_the_archive_extension(tmp_path):
    from embodify_mcp.imaging import encode_frame

    _, ext = encode_frame(np.zeros((4, 4, 3), dtype=np.uint8))
    session = make_session(tmp_path)
    session.call_tool("reset_task")
    written = sorted(p.name for p in (session.journal.dir / "images").iterdir())
    assert written == ["reset_agentview.%s" % ext, "reset_robot0_eye_in_hand.%s" % ext]


def test_protocol_images_stay_lossless_png(tmp_path):
    """Images sent to the model stay lossless PNG; lossy archiving is for replay only."""
    import base64

    session = make_session(tmp_path)
    reset = session.call_tool("reset_task")
    for item in image_blocks(reset):
        assert item["mimeType"] == "image/png"
        assert decode_png_size(base64.b64decode(item["data"])) == (32, 32)


def test_reset_without_arguments_uses_the_default_scene(tmp_path):
    """Without a choice, the first scene is used."""
    session = multi_scene_session(tmp_path)
    task = session.call_tool("reset_task")["structuredContent"]["task"]
    assert task["task_index"] == 0
    assert task["init_state_index"] == 0


def test_agent_can_pick_a_scene_by_index(tmp_path):
    session = multi_scene_session(tmp_path)
    scenes = session.call_tool("list_tasks", {"suite": "fake_suite"})["structuredContent"]["tasks"]
    assert [item["task_index"] for item in scenes] == [0, 1, 2, 3]

    chosen = session.call_tool("reset_task", {"task_index": 2})["structuredContent"]
    assert chosen["task"]["task_index"] == 2
    assert chosen["task"]["name"] == scenes[2]["name"]
    assert chosen["task"]["instruction"] == scenes[2]["instruction"]


def test_agent_can_pick_an_init_state(tmp_path):
    """init_state_index used to be fixed at 0, so a scene's layout could not change."""
    session = multi_scene_session(tmp_path)
    picked = session.call_tool("reset_task", {"task_index": 1, "init_state_index": 2})
    assert picked["structuredContent"]["task"]["init_state_index"] == 2


def test_out_of_range_scene_is_refused_with_the_valid_range(tmp_path):
    session = multi_scene_session(tmp_path)
    refused = session.call_tool("reset_task", {"task_index": 99})["structuredContent"]
    assert refused["error"] == "invalid_input"
    assert "0..3" in refused["message"]
    assert session.status == "idle", "a refusal must not leave a half-open episode"

    bad_init = session.call_tool("reset_task", {"init_state_index": 99})["structuredContent"]
    assert bad_init["error"] == "invalid_input"
    assert "init_state_index" in bad_init["message"]


def test_unknown_suite_is_refused(tmp_path):
    session = McpSession(
        FakeBackend(image_size=16, max_steps=80),
        output_root=tmp_path / "mcp",
    )  # no task_source: only its own suite is known
    refused = session.call_tool("reset_task", {"suite": "libero_goal"})["structuredContent"]
    assert refused["error"] == "invalid_input"
    assert "fake_suite" in refused["message"]


def test_lock_task_refuses_agent_side_selection_but_still_allows_a_plain_reset(tmp_path):
    """Controlled experiments must fix the scene, or the Agent could swap a hard task for an easier one."""
    session = multi_scene_session(tmp_path, locked=True, default_index=1)
    assert session.call_tool("get_session_info")["structuredContent"]["task_locked"] is True

    refused = session.call_tool("reset_task", {"task_index": 3})["structuredContent"]
    assert refused["error"] == "task_locked"

    plain = session.call_tool("reset_task")["structuredContent"]
    assert plain["task"]["task_index"] == 1, "the fixed scene is the one the server chose"


def test_journal_records_which_scene_the_run_actually_used(tmp_path):
    """Afterwards it must be clear whether the Agent switched scenes or stayed on one."""
    session = multi_scene_session(tmp_path)
    session.call_tool("reset_task", {"task_index": 3, "init_state_index": 1})
    scenes = [item for item in session.journal.read_events() if item["kind"] == "scene"]
    assert len(scenes) == 1
    assert scenes[0]["task_index"] == 3
    assert scenes[0]["init_state_index"] == 1
    assert scenes[0]["suite"] == "fake_suite"


def test_summary_records_the_scene_and_whether_it_was_locked(tmp_path):
    session = multi_scene_session(tmp_path)
    session.call_tool("reset_task", {"task_index": 2})
    stopped = session.call_tool("stop_episode")["structuredContent"]
    summary = json.loads(Path(stopped["artifacts"]["summary"]).read_text(encoding="utf-8"))
    assert summary["task"]["task_index"] == 2
    assert summary["task"]["init_state_index"] == 0
    assert summary["task_locked"] is False


def test_scene_selection_is_explained_in_the_tool_descriptions(tmp_path):
    """The Agent must know where to find the available scenes."""
    session = make_session(tmp_path)
    by_name = {tool["name"]: tool["description"] for tool in session.tools()}
    assert "list_tasks" in by_name["reset_task"], "reset_task must point to the catalogue"
    assert "task_index" in by_name["reset_task"]
    assert "init_state_index" in by_name["reset_task"]
    assert "task_locked" in by_name["get_session_info"]
    # And back: an Agent looking for the catalogue in get_session_info is pointed to list_tasks.
    assert "list_tasks" in by_name["get_session_info"]


def test_cli_exposes_the_scene_defaults_and_the_lock(tmp_path):
    from embodify_mcp.mcp import build_parser

    defaults = build_parser().parse_args([])
    assert defaults.task_index == 0
    assert defaults.init_state == 0
    assert defaults.lock_task is False
    assert build_parser().parse_args(["--lock-task"]).lock_task is True


# -- the whole benchmark is open to the Agent -----------------------------------------


def open_session(tmp_path, *, suites=("alpha", "beta"), per_suite=4):
    """A session that chooses across suites: suite_source lists them, task_source loads them on demand."""
    tasks = fake_tasks(per_suite, suite=suites[0])
    backend = FakeBackend(
        FakeSim(image_size=16, max_steps=80),
        tasks=tasks,
        task_source=lambda suite: fake_tasks(per_suite, suite=suite),
        suite_source=lambda: [{"suite": name, "n_tasks": per_suite} for name in suites],
    )
    return McpSession(backend, output_root=tmp_path / "mcp", max_steps_per_call=3)


def test_list_tasks_without_arguments_gives_only_the_suite_layer(tmp_path):
    """The first level lists only suites; pushing all 130 instructions at once would be costly."""
    session = open_session(tmp_path)
    top = session.call_tool("list_tasks")["structuredContent"]
    assert [row["suite"] for row in top["suites"]] == ["alpha", "beta"]
    assert all(row["n_tasks"] == 4 for row in top["suites"])
    assert top["n_tasks_total"] == 8
    assert "tasks" not in top, "without arguments the task list is not expanded"
    assert "instruction" not in json.dumps(top, ensure_ascii=False)


def test_list_tasks_with_a_suite_expands_that_suite(tmp_path):
    session = open_session(tmp_path)
    detail = session.call_tool("list_tasks", {"suite": "beta"})["structuredContent"]
    assert detail["suite"] == "beta"
    assert [row["task_index"] for row in detail["tasks"]] == [0, 1, 2, 3]
    assert {row["suite"] for row in detail["tasks"]} == {"beta"}
    assert all(row["instruction"] for row in detail["tasks"])


def test_listing_tasks_never_touches_the_running_episode(tmp_path):
    """Browsing the catalogue neither requires stopping the episode nor changes it."""
    session = open_session(tmp_path)
    session.call_tool("reset_task")
    session.call_tool("list_tasks")
    session.call_tool("list_tasks", {"suite": "beta"})
    info = session.call_tool("get_session_info")["structuredContent"]
    assert info["task"]["suite"] == "alpha", "listing must not change the current task"
    assert session.status == "running"


def test_session_info_no_longer_carries_the_catalogue(tmp_path):
    """Session info covers only the current run."""
    session = open_session(tmp_path)
    session.call_tool("reset_task")
    info = session.call_tool("get_session_info")["structuredContent"]
    for key in ("available_suites", "available_scenes", "scenes_of", "suites", "tasks"):
        assert key not in info, key
    assert info["task"]["suite"] == "alpha" and "state" in info


def test_every_listed_task_is_actually_startable(tmp_path):
    """The listing and the startable set must agree: listed but unstartable is worse than unlisted."""
    session = open_session(tmp_path)
    listed = []
    for entry in session.call_tool("list_tasks")["structuredContent"]["suites"]:
        listed.extend(
            session.call_tool("list_tasks", {"suite": entry["suite"]})["structuredContent"]["tasks"]
        )
    assert len(listed) == 8
    for row in listed:
        started = session.call_tool(
            "reset_task", {"suite": row["suite"], "task_index": row["task_index"]}
        )["structuredContent"]
        assert started["ok"] is True, row
        assert started["task"]["suite"] == row["suite"]
        assert started["task"]["task_index"] == row["task_index"]
        session.call_tool("stop_episode")


def test_switching_suite_without_an_index_lands_on_its_first_task(tmp_path):
    session = open_session(tmp_path)
    task = session.call_tool("reset_task", {"suite": "beta"})["structuredContent"]["task"]
    assert task["suite"] == "beta"
    assert task["task_index"] == 0


def test_unknown_suite_error_names_the_available_ones(tmp_path):
    session = open_session(tmp_path)
    refused = session.call_tool("reset_task", {"suite": "nope"})["structuredContent"]
    assert refused["error"] == "invalid_input"
    assert "alpha" in refused["message"] and "beta" in refused["message"]


def test_tool_descriptions_announce_the_full_benchmark(tmp_path):
    """The number 130 must be in the descriptions; the Agent should not have to count.

    LIBERO's texts live in LiberoBackend, which can be constructed without LIBERO installed.
    """
    from embodify_mcp.backend.libero import LiberoBackend

    session = McpSession(LiberoBackend(image_size=16), output_root=tmp_path / "mcp")
    by_name = {tool["name"]: tool for tool in session.tools()}
    reset_desc = by_name["reset_task"]["description"]
    assert "130" in reset_desc
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"):
        assert suite in reset_desc, suite
    listing = by_name["list_tasks"]
    assert "130" in listing["description"]
    assert "list_tasks" in reset_desc, "reset_task must point to list_tasks"
    assert listing["inputSchema"]["properties"]["suite"]
    assert listing["inputSchema"]["required"] == [], "without arguments it gives the suite overview"
    assert by_name["get_session_info"]["inputSchema"]["properties"] == {}, "session info takes no arguments"


def test_listing_a_suite_costs_no_simulation_steps(tmp_path):
    """Browsing the catalogue uses no step budget."""
    session = open_session(tmp_path)
    session.call_tool("reset_task")
    before = session.call_tool("get_session_info")["structuredContent"]["steps_left"]
    session.call_tool("list_tasks")
    session.call_tool("list_tasks", {"suite": "beta"})
    after = session.call_tool("get_session_info")["structuredContent"]["steps_left"]
    assert before == after


# -- task rows do not repeat the instruction --------------------------------------


def libero_shaped_tasks():
    """The naming patterns of real LIBERO tasks."""
    from embodify_mcp.sim.base import TaskSpec

    return [
        # The name is the instruction with underscores (all of libero_spatial / object / goal)
        TaskSpec(suite="probe", task_index=0,
                 name="pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
                 instruction="pick up the black bowl next to the ramekin and place it on the plate",
                 n_init_states=50),
        # Scene prefix + instruction (libero_90 / libero_10)
        TaskSpec(suite="probe", task_index=1,
                 name="KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
                 instruction="close the top drawer of the cabinet", n_init_states=50),
        TaskSpec(suite="probe", task_index=2,
                 name="LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
                 instruction="put both the alphabet soup and the tomato sauce in the basket",
                 n_init_states=50),
        # The name says something else: keep it
        TaskSpec(suite="probe", task_index=3, name="a_completely_unrelated_identifier",
                 instruction="pick up the red block", n_init_states=7),
    ]


def probe_session(tmp_path):
    tasks = libero_shaped_tasks()
    backend = FakeBackend(
        FakeSim(image_size=16, max_steps=80),
        tasks=tasks,
        task_source=lambda suite: libero_shaped_tasks(),
        suite_source=lambda: [{"suite": "probe", "n_tasks": 4}],
    )
    return McpSession(backend, output_root=tmp_path / "mcp", max_steps_per_call=3)


def test_task_rows_do_not_repeat_the_instruction_as_a_name(tmp_path):
    """LIBERO task names are the instruction with underscores; including them would double every row."""
    rows = probe_session(tmp_path).call_tool("list_tasks", {"suite": "probe"})["structuredContent"]["tasks"]
    assert "name" not in rows[0], "a name derivable from the instruction is not returned"
    assert "scene" not in rows[0], "no scene is invented without a prefix"
    assert rows[0]["instruction"].startswith("pick up the black bowl")


def test_scene_prefix_is_kept_because_it_is_new_information(tmp_path):
    """The same "close the drawer" appears in several scenes; the prefix tells them apart."""
    rows = probe_session(tmp_path).call_tool("list_tasks", {"suite": "probe"})["structuredContent"]["tasks"]
    assert rows[1]["scene"] == "KITCHEN_SCENE10"
    assert "name" not in rows[1], "the rest after the prefix repeats the instruction"
    assert rows[2]["scene"] == "LIVING_ROOM_SCENE2", "room names containing underscores are recognized"


def test_a_name_that_says_something_else_is_kept(tmp_path):
    """Better too much than lost information: a name that cannot be derived is returned as is."""
    rows = probe_session(tmp_path).call_tool("list_tasks", {"suite": "probe"})["structuredContent"]["tasks"]
    assert rows[3]["name"] == "a_completely_unrelated_identifier"
    assert rows[3]["n_init_states"] == 7


def test_a_row_without_a_name_is_still_enough_to_start_the_task(tmp_path):
    """Omitting the name does not affect usability: reset_task takes the task_index."""
    session = probe_session(tmp_path)
    row = session.call_tool("list_tasks", {"suite": "probe"})["structuredContent"]["tasks"][0]
    assert "name" not in row
    started = session.call_tool(
        "reset_task", {"suite": row["suite"], "task_index": row["task_index"]}
    )["structuredContent"]
    assert started["ok"] is True
    assert started["task"]["instruction"] == row["instruction"]


class ScriptedSuccess:
    """A simulator whose success the test controls.

    `FakeSim.success()` needs the object to really be in the goal zone; moving it there to test success
    bookkeeping would make the test depend on FakeSim's dynamics. This wrapper takes over `success()`
    and passes everything else through.
    """

    def __init__(self, inner, *, succeed_after=None, undo_after=None, raises=False):
        self.inner = inner
        self.succeed_after = succeed_after
        self.undo_after = undo_after
        self.raises = raises
        self.queries = 0

    def success(self):
        self.queries += 1
        if self.raises:
            raise RuntimeError("this simulator cannot answer")
        if self.succeed_after is None:
            return False
        step = self.inner.observe().step_index
        if self.undo_after is not None and step >= self.undo_after:
            return False
        return step >= self.succeed_after

    def __getattr__(self, name):
        return getattr(self.inner, name)


def run_until_stopped(session, *, moves=4):
    """Run a few moves, then stop; returns the stop result, the summary and the events."""
    session.call_tool("reset_task")
    for _ in range(moves):
        session.call_tool("move_relative", {"delta_xyz": FAR_AWAY})
    stopped = session.call_tool("stop_episode")
    summary = json.loads(Path(stopped["structuredContent"]["artifacts"]["summary"]).read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in Path(stopped["structuredContent"]["artifacts"]["events"]).read_text(encoding="utf-8").splitlines()
        if line
    ]
    return stopped, summary, events


def scripted_session(tmp_path, **kwargs):
    sim = ScriptedSuccess(FakeSim(image_size=16, max_steps=80), **kwargs)
    return McpSession(FakeBackend(sim), output_root=tmp_path / "mcp", max_steps_per_call=3)


def test_summary_records_success_and_the_step_it_first_held(tmp_path):
    """Success is written in stop_episode: after close() nobody can ask anymore."""
    session = scripted_session(tmp_path, succeed_after=5)
    _stopped, summary, events = run_until_stopped(session)
    assert summary["success"] is True
    assert summary["success_at_step"] == 5
    assert summary["success_at_stop"] is True
    # Only the first time is recorded
    marks = [item for item in events if item["kind"] == "success"]
    assert len(marks) == 1 and marks[0]["step"] == 5


def test_stop_never_tells_the_agent_whether_it_succeeded(tmp_path):
    """Information boundary: success is written to disk but never returned to the Agent."""
    session = scripted_session(tmp_path, succeed_after=1)
    stopped, summary, _events = run_until_stopped(session)
    assert summary["success"] is True
    assert "success" not in json.dumps(stopped)


def test_a_failed_episode_is_recorded_as_a_failure(tmp_path):
    session = scripted_session(tmp_path)
    _stopped, summary, events = run_until_stopped(session)
    assert summary["success"] is False
    assert summary["success_at_step"] is None
    assert summary["success_at_stop"] is False
    assert not [item for item in events if item["kind"] == "success"]


def test_a_success_that_gets_undone_is_still_a_success(tmp_path):
    """LIBERO re-checks its BDDL predicates every step; a bowl placed and then knocked off still counts as solved."""
    session = scripted_session(tmp_path, succeed_after=2, undo_after=6)
    _stopped, summary, _events = run_until_stopped(session)
    assert summary["success"] is True, "solved once means solved"
    assert summary["success_at_step"] == 2
    assert summary["success_at_stop"] is False, "it no longer holds at the end, and that is recorded too"


def test_a_simulator_that_cannot_answer_gives_null_not_false(tmp_path):
    """Unknown and failed are different; mixing them would understate success rates."""
    session = scripted_session(tmp_path, raises=True)
    stopped, summary, events = run_until_stopped(session)
    assert summary["success"] is None
    assert summary["success_at_stop"] is None
    assert stopped["structuredContent"]["status"] == "stopped", "an unanswerable success query must not break the episode"
    # A broken probe is reported once, not once per step
    assert len([item for item in events if item["kind"] == "success_unavailable"]) == 1


def test_success_does_not_carry_over_to_the_next_episode(tmp_path):
    """A second episode in the same server must not inherit the first one's result."""
    sim = ScriptedSuccess(FakeSim(image_size=16, max_steps=80), succeed_after=1)
    session = McpSession(FakeBackend(sim), output_root=tmp_path / "mcp", max_steps_per_call=3)
    session.call_tool("reset_task")
    session.call_tool("move_relative", {"delta_xyz": FAR_AWAY})
    first = json.loads(
        Path(session.call_tool("stop_episode")["structuredContent"]["artifacts"]["summary"]).read_text(encoding="utf-8")
    )
    assert first["success"] is True

    sim.succeed_after = None
    session.call_tool("reset_task")
    session.call_tool("move_relative", {"delta_xyz": FAR_AWAY})
    second = json.loads(
        Path(session.call_tool("stop_episode")["structuredContent"]["artifacts"]["summary"]).read_text(encoding="utf-8")
    )
    assert second["success"] is False
    assert second["success_at_step"] is None


# -- every tool call goes to the journal (the monitor and replay match frames to calls with it) --


def journal_events(session):
    path = session.journal.dir / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_every_call_of_an_episode_is_journaled_in_order(tmp_path):
    session = make_session(tmp_path)
    session.call_tool("reset_task")
    session.call_tool("observe")
    session.call_tool("move_relative", {"delta_xyz": [0.01, 0.0, 0.0]})
    session.call_tool("set_gripper", {"state": "close"})
    session.call_tool("get_session_info")
    session.call_tool("stop_episode")
    calls = [e for e in journal_events(session) if e["kind"] == "tool_call"]
    assert [c["name"] for c in calls] == [
        "reset_task", "observe", "move_relative", "set_gripper", "get_session_info", "stop_episode",
    ]
    assert all(c["ok"] for c in calls)
    move = calls[2]
    assert move["arguments"] == {"delta_xyz": [0.01, 0.0, 0.0]}
    assert move["step_to"] > move["step_from"]
    assert move["result"]["action"]["stop_reason"]
    assert move["duration_s"] >= 0


def test_frames_of_a_call_come_before_its_tool_call_event(tmp_path):
    """Replay assigns the frames between the previous tool_call and this one to this call, so order matters."""
    session = make_session(tmp_path)
    session.call_tool("reset_task")
    session.call_tool("move_relative", {"delta_xyz": FAR_AWAY})
    kinds = [(e["kind"], e.get("label") or e.get("name")) for e in journal_events(session)]
    first_call = kinds.index(("tool_call", "reset_task"))
    assert kinds[first_call - 1] == ("observation", "reset")
    moves = [i for i, k in enumerate(kinds) if k[0] == "observation" and k[1].startswith("move_")]
    assert moves and max(moves) < kinds.index(("tool_call", "move_relative"))
    assert min(moves) > first_call


def test_failed_calls_are_journaled_too(tmp_path):
    session = make_session(tmp_path)
    session.call_tool("reset_task")
    session.call_tool("set_gripper", {"state": "sideways"})
    call = [e for e in journal_events(session) if e["kind"] == "tool_call"][-1]
    assert call["ok"] is False
    assert call["error"] == "invalid_input"
    assert call["step_from"] == call["step_to"]


def test_calls_between_episodes_are_not_pinned_on_the_finished_one(tmp_path):
    session = make_session(tmp_path)
    session.call_tool("reset_task")
    session.call_tool("stop_episode")
    finished = journal_events(session)
    session.call_tool("list_tasks")
    session.call_tool("get_session_info")
    assert journal_events(session) == finished


def test_the_catalogue_is_not_copied_into_the_journal(tmp_path):
    session = make_session(tmp_path)
    session.call_tool("reset_task")
    session.call_tool("list_tasks")
    call = [e for e in journal_events(session) if e["kind"] == "tool_call"][-1]
    assert call["name"] == "list_tasks"
    assert isinstance(call["result"]["suites"], str)


def test_scene_event_names_the_server_launch(tmp_path):
    """Every reset gets a new timestamp directory; session groups the episodes of one server launch."""
    session = make_session(tmp_path)
    session.call_tool("reset_task")
    first = session.journal.dir
    session.call_tool("stop_episode")
    session.call_tool("reset_task")
    scenes = [
        json.loads(line)
        for d in (first, session.journal.dir)
        for line in (d / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if '"scene"' in line
    ]
    assert len(scenes) == 2
    assert scenes[0]["session"] == scenes[1]["session"] == session.session_id
    assert scenes[0]["instruction"]


# -- input validation ------------------------------------------------------------------


def test_wrongly_typed_input_is_invalid_input_not_a_simulator_error(tmp_path):
    """A displacement of "abc" used to be reported as sim_error, so the Agent blamed the simulator for its own mistake."""
    session = make_session(tmp_path)
    session.call_tool("reset_task")
    for args in ({"delta_xyz": "abc"}, {"delta_xyz": [0, 0, "x"]}, {"delta_xyz": {"x": 1}},
                 {"delta_xyz": [0, 0, 0], "delta_rpy": "yaw"}):
        refused = session.call_tool("move_relative", args)["structuredContent"]
        assert refused["error"] == "invalid_input", args
    refused = session.call_tool("set_gripper", {"state": ["close"]})["structuredContent"]
    assert refused["error"] == "invalid_input"


def test_arguments_that_are_not_an_object_are_refused(tmp_path):
    session = make_session(tmp_path)
    refused = session.call_tool("reset_task", ["task_index", 1])
    assert refused["isError"] is True
    assert refused["structuredContent"]["error"] == "invalid_input"
