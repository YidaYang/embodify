"""P2 local transport integration and fault injection; no remote machine required."""
import base64
import json
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from embodify_mcp.backend.base import BackendDisconnected, ExtraResult, McpToolError
from embodify_mcp.backend.fake import FakeBackend
from embodify_mcp.backend.remote import RemoteBackend
from embodify_mcp.backend.serve import serve_connection
from embodify_mcp.backend.transport import socket_transport
from embodify_mcp.backend.wire import codecs, decode_image, dumps, encode_image, loads
from embodify_mcp.mcp import McpSession

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def service(backend=None, **client_options):
    backend = backend or FakeBackend(image_size=16, max_steps=80)
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(2)
    listener.settimeout(0.1)
    stopped = threading.Event()
    connections = []
    errors = []

    def run():
        try:
            while not stopped.is_set():
                try:
                    sock, _ = listener.accept()
                except socket.timeout:
                    continue
                connection = socket_transport(sock)
                connections.append(connection)
                serve_connection(backend, connection, heartbeat_interval=0.02, peer_timeout=3)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    remote = None
    try:
        remote = RemoteBackend(host='127.0.0.1', port=listener.getsockname()[1], **client_options)
        yield remote, connections
    finally:
        if remote:
            remote.close()
        stopped.set()
        for connection in connections:
            connection.close()
        worker.join(5)
        listener.close()
        assert not worker.is_alive(), 'backend service did not release its owner'
        assert not errors


def start(remote):
    task, variant = remote.catalogue.resolve({}, locked=False)
    remote.catalogue.commit(task, variant)
    return remote.reset(task, variant)


@pytest.mark.parametrize('codec', codecs())
def test_image_roundtrip_preserves_agent_pixels(codec):
    image = np.random.RandomState(7).randint(0, 256, (29, 31, 3), dtype=np.uint8)
    assert np.array_equal(image, decode_image(encode_image(image, codec)))


@pytest.mark.parametrize('mutation', ['shape', 'base64', 'truncated', 'expanded', 'codec'])
def test_reject_invalid_images(mutation):
    value = encode_image(np.zeros((3, 4, 3), dtype=np.uint8), 'rgb-zlib')
    if mutation == 'shape':
        value['shape'] = [99999999, 4, 3]
    elif mutation == 'base64':
        value['data'] = '!!!'
    elif mutation == 'truncated':
        value['data'] = base64.b64encode(base64.b64decode(value['data'])[:-2]).decode()
    elif mutation == 'expanded':
        value['shape'] = [1, 1, 3]
    else:
        value['codec'] = 'pickle'
    with pytest.raises(ValueError):
        decode_image(value)


@pytest.mark.parametrize('data', [b'{"v":2}', b'[]', b'{"v":1,"x":NaN}'])
def test_reject_invalid_envelope(data):
    with pytest.raises(ValueError):
        loads(data)


class PulseBackend(FakeBackend):
    def success(self):
        return self.observe().step == 2


def test_stride_retains_each_state_and_transient_success_locally(tmp_path):
    with service(PulseBackend(image_size=16), frame_stride=3) as (remote, _):
        session = McpSession(remote, output_root=tmp_path, max_steps_per_call=5)
        assert not session.call_tool('reset_task').get('isError')
        moved = session.call_tool('move_relative', {'delta_xyz': [10, 0, 0]})
        assert not moved.get('isError')
        assert len([x for x in moved['content'] if x['type'] == 'image']) == 2
        events = session.journal.read_events()
        frames = [x for x in events if x['kind'] == 'observation' and x['step'] > 0]
        assert [x['step'] for x in frames] == [1, 2, 3, 4, 5]
        assert [x['step'] for x in frames if x['image_cameras']] == [1, 3, 5]
        assert [x['step'] for x in events if x['kind'] == 'success'] == [2]
        metric = [x for x in events if x['kind'] == 'tool_call'][-1]['transport'][-1]
        assert metric['progress_steps'] == 5 and metric['received_bytes'] > 0
        assert metric['roundtrip_s'] >= metric['server_s'] > 0
        stopped = session.call_tool('stop_episode')
        assert 'success' not in json.dumps([moved, stopped])
        summary = json.loads((session.journal.dir / 'summary.json').read_text())
        assert summary['success'] is True and summary['success_at_stop'] is False


class SlowBackend(FakeBackend):
    def __init__(self):
        super().__init__(image_size=16)
        self.delay = 0.2
        self.calls = 0

    def move_to(self, *args, **kwargs):
        self.calls += 1
        time.sleep(self.delay)
        return super().move_to(*args, **kwargs)


def test_heartbeats_keep_a_slow_operation_alive():
    backend = SlowBackend()
    with service(backend, heartbeat_timeout_s=0.1, timeout_s=2) as (remote, _):
        snap = start(remote)
        arm = remote.info.arm_names[0]
        state = snap.arms[arm]
        report = remote.move_to(arm, state.pos + [0, 0, .1], state.quat, max_steps=1)
        assert report.executed_steps == 1
        assert backend.calls == 1


def test_timeout_aborts_without_retry_and_requires_reset(tmp_path):
    backend = SlowBackend()
    with service(backend, timeout_s=2) as (remote, _):
        session = McpSession(remote, output_root=tmp_path)
        assert not session.call_tool('reset_task').get('isError')
        remote.timeout_s = .05
        result = session.call_tool('move_relative', {'delta_xyz': [0, 0, .1]})
        assert result['structuredContent']['error'] == 'remote_timeout'
        assert session.status == 'aborted'
        summary = json.loads((session.journal.dir / 'summary.json').read_text())
        assert summary['status'] == 'aborted' and summary['success'] is None
        assert session.call_tool('move_relative', {'delta_xyz': [0, 0, .1]})['isError']
        assert backend.calls == 1
        remote.timeout_s = 2
        assert not session.call_tool('reset_task').get('isError')
        assert session.step_index == 0
        backend.delay = 0
        assert not session.call_tool('move_relative', {'delta_xyz': [0, 0, .1]}).get('isError')
        assert backend.calls == 2
        session.close()


def test_disconnect_preserves_log_and_reconnects_as_new_episode(tmp_path):
    with service() as (remote, connections):
        session = McpSession(remote, output_root=tmp_path)
        session.call_tool('reset_task')
        old = session.journal.dir
        connections[0].close()
        result = session.call_tool('observe')
        assert result['structuredContent']['error'] == 'remote_disconnected'
        assert session.status == 'aborted'
        assert list((old / 'images').iterdir())
        with pytest.raises(McpToolError, match='reset_task'):
            remote.observe()
        assert not session.call_tool('reset_task').get('isError')
        assert session.journal.dir != old and session.step_index == 0
        session.close()
        assert json.loads((session.journal.dir / 'summary.json').read_text())['reason'] == 'host_disconnected'


class FailingReset(FakeBackend):
    def reset(self, *args):
        raise RuntimeError('private server pathname should not reach the Agent')


def test_failed_reset_is_archived_and_sanitized(tmp_path):
    with service(FailingReset()) as (remote, _):
        session = McpSession(remote, output_root=tmp_path)
        result = session.call_tool('reset_task')
        assert result['structuredContent']['error'] == 'remote_backend_error'
        assert 'pathname' not in json.dumps(result)
        assert session.status == 'aborted'
        assert session.journal.read_events()[-1]['kind'] == 'tool_call'
        assert json.loads((session.journal.dir / 'summary.json').read_text())['steps'] == 0


class ExtendedBackend(FakeBackend):
    def extra_tools(self):
        return [{'name': 'inspect_attachment', 'description': 'demo', 'inputSchema': {'type': 'object'}}]

    def call_extra(self, name, args, *, running):
        return ExtraResult({'name': name, 'running': running}, [np.zeros((4, 4, 3), dtype=np.uint8)], {'audit': 5})

    def episode_audit(self):
        return {'physics': 'private audit'}


def test_extra_tools_and_audit_are_forwarded(tmp_path):
    with service(ExtendedBackend()) as (remote, _):
        session = McpSession(remote, output_root=tmp_path)
        assert session.tools()[-1]['name'] == 'inspect_attachment'
        assert not session.call_tool('inspect_attachment').get('isError')
        session.call_tool('reset_task')
        result = session.call_tool('inspect_attachment')
        assert any(x['type'] == 'image' for x in result['content'])
        stopped = session.call_tool('stop_episode')
        assert 'private audit' not in json.dumps(stopped)
        assert json.loads((session.journal.dir / 'summary.json').read_text())['physics_audit']['physics'] == 'private audit'


def test_config_and_full_mcp_stdio_chain(tmp_path):
    config = tmp_path / 'backend.json'
    config.write_text(json.dumps({'transport': 'stdio', 'command': [sys.executable, '-m',
        'embodify_mcp.backend.serve', '--backend', 'fake', '--image-size', '16'], 'cwd': str(ROOT)}))
    commands = [{'jsonrpc': '2.0', 'id': i, 'method': 'tools/call', 'params': {'name': name, 'arguments': args}}
        for i, (name, args) in enumerate([('reset_task', {}), ('move_relative', {'delta_xyz': [0, 0, .02]}),
                                         ('set_gripper', {'state': 'close'}), ('stop_episode', {})], 1)]
    completed = subprocess.run([sys.executable, '-m', 'embodify_mcp.mcp', '--backend', 'remote',
        '--remote-config', str(config), '--output-root', str(tmp_path / 'logs')], cwd=str(ROOT),
        input=''.join(json.dumps(x) + '\n' for x in commands), encoding='utf-8',
        capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    results = [json.loads(x) for x in completed.stdout.splitlines()]
    assert len(results) == 4
    assert all(not x['result'].get('isError') for x in results), results
    summary, = (tmp_path / 'logs').glob('*/r1/summary.json')
    assert json.loads(summary.read_text())['status'] == 'stopped'

@pytest.mark.parametrize('fault', ['wrong_id', 'skipped_step', 'bad_count', 'missing_camera'])
def test_malformed_responses_abort_the_episode(tmp_path, fault):
    with service() as (remote, connections):
        session = McpSession(remote, output_root=tmp_path, max_steps_per_call=3)
        session.call_tool('reset_task')
        connection = connections[0]
        original = connection.send

        def corrupt(value, **kwargs):
            if value['kind'] == 'progress':
                if fault == 'wrong_id':
                    value['id'] += 100
                if fault == 'skipped_step':
                    value['snapshot']['step'] += 1
            if value['kind'] == 'result' and 'executed_steps' in value['result']:
                if fault == 'bad_count':
                    value['result']['executed_steps'] += 1
                if fault == 'missing_camera':
                    value['result']['snapshot']['images'] = {}
            return original(value, **kwargs)

        connection.send = corrupt
        result = session.call_tool('move_relative', {'delta_xyz': [10, 0, 0]})
        assert result['structuredContent']['error'] == 'remote_protocol_error'
        assert session.status == 'aborted'


def test_disconnect_in_motion_preserves_received_steps(tmp_path):
    from embodify_mcp.backend.transport import TransportError
    with service() as (remote, connections):
        session = McpSession(remote, output_root=tmp_path, max_steps_per_call=5)
        session.call_tool('reset_task')
        connection = connections[0]
        original = connection.send

        def cut(value, **kwargs):
            if value['kind'] == 'progress' and value['snapshot']['step'] == 3:
                connection.close()
                raise TransportError('injected disconnect')
            return original(value, **kwargs)

        connection.send = cut
        result = session.call_tool('move_relative', {'delta_xyz': [10, 0, 0]})
        assert result['isError'] and session.status == 'aborted'
        summary = json.loads((session.journal.dir / 'summary.json').read_text())
        assert summary['steps'] == 2
        frames = [e for e in session.journal.read_events() if e['kind'] == 'observation']
        assert [e['step'] for e in frames] == [0, 1, 2]


def test_heartbeat_loss_interrupts_a_silent_operation(tmp_path):
    backend = SlowBackend()
    backend.delay = .4
    with service(backend, heartbeat_timeout_s=.15, timeout_s=2) as (remote, connections):
        session = McpSession(remote, output_root=tmp_path)
        session.call_tool('reset_task')
        original = connections[0].send
        connections[0].send = lambda value, **kw: None if value['kind'] == 'heartbeat' else original(value, **kw)
        result = session.call_tool('move_relative', {'delta_xyz': [10, 0, 0]})
        assert result['structuredContent']['error'] == 'remote_disconnected'
        assert session.status == 'aborted' and backend.calls == 1


def test_tcp_service_cli_accepts_sequential_owners():
    import queue
    proc = subprocess.Popen([sys.executable, '-m', 'embodify_mcp.backend.serve', '--backend', 'fake',
        '--image-size', '16', '--listen-port', '0'], cwd=str(ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    lines = queue.Queue()
    def read_stderr():
        for line in proc.stderr:
            lines.put(line)
    threading.Thread(target=read_stderr, daemon=True).start()
    try:
        line = lines.get(timeout=10)
        assert line.startswith(b'BACKEND_READY '), line
        address = json.loads(line.split(b' ', 1)[1])
        for _ in range(2):
            remote = RemoteBackend(**address)
            try:
                assert start(remote).step == 0
            finally:
                remote.close()
        assert proc.poll() is None
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        assert proc.stdout.read() == b'', 'backend stdout must contain only protocol messages'
        proc.stdout.close()
        proc.stderr.close()

def test_negotiated_rgb_fallback_and_invalid_task_keep_connection(monkeypatch):
    import embodify_mcp.backend.remote as module
    monkeypatch.setattr(module, 'codecs', lambda: ['rgb-zlib'])
    with service() as (remote, _):
        assert remote.journal_params()['remote']['codec'] == 'rgb-zlib'
        with pytest.raises(McpToolError):
            remote.catalogue.resolve({'task_index': -1}, locked=False)
        assert start(remote).images
        assert remote._connection is not None


class FailedCleanup(FakeBackend):
    def episode_audit(self):
        raise RuntimeError('injected audit failure')


def test_failed_stop_never_publishes_successful_finish(tmp_path):
    with service(FailedCleanup()) as (remote, _):
        session = McpSession(remote, output_root=tmp_path)
        session.call_tool('reset_task')
        result = session.call_tool('stop_episode')
        assert result['isError'] and session.status == 'aborted'
        assert json.loads((session.journal.dir / 'summary.json').read_text())['status'] == 'aborted'
        assert len([e for e in session.journal.read_events() if e['kind'] == 'finish']) == 1
