import json
import threading
import urllib.error
import urllib.request

import pytest

from embodify_mcp.backend.fake import FakeBackend
from embodify_mcp.mcp import McpSession
from embodify_mcp.monitor import RunIndex, make_server, read_events


@pytest.fixture
def served(tmp_path):
    """A short episode plus a monitor server pointing at it."""
    root = tmp_path / "mcp"
    session = McpSession(FakeBackend(image_size=16, max_steps=80), output_root=root, max_steps_per_call=3)
    session.call_tool("reset_task")
    session.call_tool("move_relative", {"delta_xyz": [0.02, 0.0, 0.0]})
    server = make_server(root, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    run_id = f"{session.journal.dir.parent.name}/{session.journal.dir.name}"
    yield session, base, run_id
    server.shutdown()
    server.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as res:
        return res.status, res.headers.get("Content-Type"), res.read()


def get_json(url):
    return json.loads(get(url)[2])


def test_page_is_served(served):
    _, base, _ = served
    status, mime, body = get(base + "/")
    assert status == 200 and mime.startswith("text/html")
    assert "Embodify Monitor" in body.decode("utf-8")


def test_running_episode_is_listed_as_unfinished(served):
    session, base, run_id = served
    runs = get_json(base + "/api/runs")["runs"]
    assert [r["id"] for r in runs] == [run_id]
    assert runs[0]["finished"] is False
    assert runs[0]["session"] == session.session_id
    assert runs[0]["instruction"]


def test_events_can_be_tailed_while_the_episode_runs(served):
    session, base, run_id = served
    first = get_json(f"{base}/api/run/{run_id}/events?offset=0")
    assert first["manifest"]["run_id"] == "r1"
    assert first["summary"] is None
    kinds = [e["kind"] for e in first["events"]]
    assert "observation" in kinds and "tool_call" in kinds
    session.call_tool("observe")
    session.call_tool("stop_episode")
    more = get_json(f"{base}/api/run/{run_id}/events?offset={first['offset']}")
    assert [e.get("name") for e in more["events"] if e["kind"] == "tool_call"] == ["observe", "stop_episode"]
    assert "manifest" not in more  # only the first read needs it
    assert more["summary"]["status"] == "stopped"
    assert more["offset"] == more["size"]


def test_every_observation_has_both_camera_frames(served):
    _, base, run_id = served
    events = get_json(f"{base}/api/run/{run_id}/events?offset=0")["events"]
    labels = [e["label"] for e in events if e["kind"] == "observation"]
    assert labels
    for label in labels:
        for camera in ("agentview", "robot0_eye_in_hand"):
            status, mime, body = get(f"{base}/api/run/{run_id}/frame/{label}/{camera}")
            assert status == 200 and mime.startswith("image/") and body


@pytest.mark.parametrize(
    "path",
    [
        "/api/run/..%2F..%2Fetc/r1/events",
        "/api/run/x/r1/frame/..%2Fmanifest/agentview",
        "/api/run/nope/r1/events",
        "/api/nothing",
    ],
)
def test_paths_outside_the_output_root_are_refused(served, path):
    _, base, _ = served
    with pytest.raises(urllib.error.HTTPError) as err:
        get(base + path)
    assert err.value.code == 404


def test_a_half_written_line_waits_for_the_next_read(tmp_path):
    """While MCP appends, the last line may be half written; it must not be dropped as a bad line."""
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"kind": "a"}\n{"kind": "b", "x": ')
    events, offset = read_events(path)
    assert events == [{"kind": "a"}] and offset == 14
    with path.open("ab") as handle:
        handle.write(b"1}\n")
    events, offset = read_events(path, offset)
    assert events == [{"kind": "b", "x": 1}] and offset == path.stat().st_size


def test_nan_does_not_break_the_browser(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"kind": "mcp_move", "remaining_distance_m": NaN}\n', encoding="utf-8")
    events, _ = read_events(path)
    assert events[0]["remaining_distance_m"] is None
    json.dumps(events, allow_nan=False)


def test_old_runs_without_scene_info_still_list(tmp_path):
    run = tmp_path / "20260101-000000" / "r1"
    run.mkdir(parents=True)
    (run / "events.jsonl").write_text('{"t": 1, "kind": "observation", "label": "reset", "step": 0}\n', encoding="utf-8")
    (run / "summary.json").write_text('{"status": "stopped", "steps": 0}', encoding="utf-8")
    [info] = RunIndex(tmp_path).list_runs()
    assert info["finished"] is True and info["success"] is None and info["session"] is None
