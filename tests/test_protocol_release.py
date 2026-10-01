"""Regression cases for the public distribution's protocol boundary."""
import io
import json
import pytest
from embodify_mcp.backend.fake import FakeBackend
from embodify_mcp.mcp import McpSession, rpc_response, serve_stdio, MAX_REQUEST_CHARS

@pytest.fixture
def session(tmp_path):
    s = McpSession(FakeBackend(image_size=16, max_steps=10), output_root=tmp_path)
    yield s
    s.close()


def test_unknown_protocol_version_negotiates_supported_version(session):
    result = rpc_response({"jsonrpc":"2.0", "id":1, "method":"initialize",
                           "params":{"protocolVersion":"not-a-real-version"}}, session)
    assert result["result"]["protocolVersion"] == "2025-11-25"

@pytest.mark.parametrize("params", [[], [1], None, "bad", {"name":"observe", "arguments":[]}])
def test_invalid_params_preserve_request_id(session, params):
    r = rpc_response({"jsonrpc":"2.0", "id":"req", "method":"tools/call", "params":params}, session)
    assert r["id"] == "req"
    assert r["error"]["code"] == -32602
    assert session.status == "idle"

@pytest.mark.parametrize("envelope", [[], None, {"method":"ping", "id":1},
    {"jsonrpc":"2.0", "method":"ping", "id":True}])
def test_invalid_envelopes(session, envelope):
    assert rpc_response(envelope, session)["error"]["code"] == -32600


def test_notification_cannot_execute_reset(session):
    assert rpc_response({"jsonrpc":"2.0", "method":"tools/call",
                        "params":{"name":"reset_task"}}, session) is None
    assert session.status == "idle"


def test_parse_error_does_not_break_following_request(session):
    stream = io.StringIO('not-json\n{"jsonrpc":"2.0","id":9,"method":"ping"}\n')
    sink = io.StringIO()
    serve_stdio(session, stream, sink)
    replies = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert replies[0]["error"]["code"] == -32700
    assert replies[1] == {"jsonrpc":"2.0", "id":9, "result":{}}


def test_oversized_request_does_not_dispatch_tail(session):
    sink = io.StringIO()
    tail = json.dumps({"jsonrpc":"2.0", "id":1, "method":"tools/call", "params":{"name":"reset_task"}})
    serve_stdio(session, io.StringIO(" " * (MAX_REQUEST_CHARS+1) + "\n" + tail + "\n"), sink)
    assert len(sink.getvalue().splitlines()) == 1
    assert session.status == "idle"


def test_unexpected_backend_error_is_not_returned_verbatim(session, monkeypatch):
    def explode(*args):
        raise RuntimeError("operator-private-data")
    monkeypatch.setattr(session, "observe", explode)
    result = session.call_tool("observe")
    assert result["isError"]
    assert "operator-private-data" not in json.dumps(result)
