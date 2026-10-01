"""Latency/jitter regression tests over real loopback sockets, without remote access."""
import queue
import socket
import threading
import time
from contextlib import contextmanager

import pytest

from embodify_mcp.backend.fake import FakeBackend
from embodify_mcp.backend.remote import RemoteBackend
from embodify_mcp.backend.transport import socket_transport
from embodify_mcp.backend.wire import dumps
from embodify_mcp.mcp import McpSession
from tests.test_remote_backend import service


@contextmanager
def delayed_link(host, port):
    """Ordered full-duplex byte relay: 1.05..1.15 s ONE-WAY delay, pipelined, no per-line ACK."""
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    stop = threading.Event()
    sockets = [listener]
    workers = []
    errors = []

    def relay(source, destination):
        packets = queue.Queue()
        def transmit():
            try:
                while not stop.is_set():
                    try:
                        due, data = packets.get(timeout=.05)
                    except queue.Empty:
                        continue
                    if stop.wait(max(0, due - time.monotonic())):
                        return
                    destination.sendall(data)
            except OSError:
                pass  # Closing either endpoint is normal teardown.
        tx = threading.Thread(target=transmit, daemon=True)
        workers.append(tx)
        tx.start()
        n, last_due = 0, 0
        try:
            while not stop.is_set():
                data = source.recv(65536)
                if not data:
                    return
                due = max(last_due, time.monotonic() + 1.05 + (n % 3) * .05)
                packets.put((due, data))
                last_due, n = due, n + 1
        except OSError:
            pass

    def accept():
        try:
            client, _ = listener.accept()
            server = socket.create_connection((host, port), timeout=5)
            sockets.extend([client, server])
            for source, destination in [(client, server), (server, client)]:
                source.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                worker = threading.Thread(target=relay, args=(source, destination), daemon=True)
                workers.append(worker)
                worker.start()
        except OSError as exc:
            if not stop.is_set():
                errors.append(exc)

    acceptor = threading.Thread(target=accept, daemon=True)
    acceptor.start()
    try:
        yield listener.getsockname()
    finally:
        stop.set()
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        acceptor.join(3)
        for worker in workers:
            worker.join(3)
            assert not worker.is_alive()
        assert not acceptor.is_alive() and not errors


class CountMoves(FakeBackend):
    def __init__(self):
        super().__init__(image_size=16)
        self.moves = 0
        self.motion_duration = None

    def move_to(self, *args, **kwargs):
        self.moves += 1
        before = time.monotonic()
        result = super().move_to(*args, **kwargs)
        self.motion_duration = time.monotonic() - before
        return result


def test_two_second_rtt_with_jitter_completes_episode_without_step_roundtrips(tmp_path):
    backend = CountMoves()
    with service(backend) as (initial, _):
        initial.close()  # Release the service's initial owner before connecting through the relay.
        with delayed_link(initial.host, initial.port) as (host, port):
            remote = RemoteBackend(host=host, port=port, frame_stride=5)
            try:
                session = McpSession(remote, output_root=tmp_path, max_steps_per_call=30)
                assert not session.call_tool('reset_task').get('isError')
                before = time.monotonic()
                result = session.call_tool('move_relative', {'delta_xyz': [10, 0, 0]})
                elapsed = time.monotonic() - before
                assert not result.get('isError'), result
                assert result['structuredContent']['action']['executed_steps'] == 30
                assert backend.moves == 1
                # observe + move need two RPCs; 30 physics steps must not cost 30 RTTs.
                assert 4 <= elapsed < 15
                assert backend.motion_duration < 3
                events = session.journal.read_events()
                frames = [e for e in events if e['kind'] == 'observation']
                assert [e['step'] for e in frames] == list(range(31))
                assert not session.call_tool('set_gripper', {'state': 'close'}).get('isError')
                assert not session.call_tool('stop_episode').get('isError')
                assert session.status == 'stopped'
                remote.drain_transport_events()
                remote.close()
                closed = remote.drain_transport_events()
                assert closed[-1]['method'] == 'close' and closed[-1]['error'] is None
                assert closed[-1]['roundtrip_s'] >= 2
            finally:
                remote.close()


def test_fragmented_image_data_counts_as_liveness_before_newline():
    raw, incoming = socket.socketpair()
    transport = socket_transport(incoming)
    transport.keepalive(.02, .15)
    payload = dumps({'v': 1, 'kind': 'result', 'id': 1, 'image': 'a' * 10000})
    errors = []
    def trickle():
        try:
            for i in range(0, len(payload), 500):
                raw.sendall(payload[i:i+500])
                time.sleep(.035)
        except OSError as exc:
            errors.append(exc)
    sender = threading.Thread(target=trickle, daemon=True)
    sender.start()
    try:
        reply = transport.receive(3)
        assert reply['image'] == 'a' * 10000
        assert not transport.closed.is_set()
        assert transport.bytes_received == len(payload)
    finally:
        sender.join(3)
        transport.close()
        raw.close()
    assert not errors


def test_heartbeat_does_not_timeout_behind_an_existing_data_write():
    left, right = socket.socketpair()
    sender, receiver = socket_transport(left), socket_transport(right)
    entered, release = threading.Event(), threading.Event()
    original = sender.writer
    errors = []
    class SlowWriter:
        def write(self, data):
            entered.set()
            assert release.wait(3)
            return original.write(data)
        def flush(self):
            original.flush()
    sender.writer = SlowWriter()
    def send():
        try:
            sender.send({'kind': 'result', 'id': 1}, timeout=2)
        except Exception as exc:
            errors.append(exc)
    worker = threading.Thread(target=send, daemon=True)
    worker.start()
    try:
        assert entered.wait(1)
        receiver.keepalive(.02, 2)
        sender.keepalive(.02, .15)
        time.sleep(.35)
        assert not sender.closed.is_set()
        release.set()
        assert receiver.receive(2)['id'] == 1
    finally:
        release.set()
        worker.join(3)
        sender.close()
        receiver.close()
    assert not errors


def test_slow_but_live_stream_still_obeys_operation_deadline(tmp_path):
    with service(heartbeat_timeout_s=.15) as (remote, connections):
        session = McpSession(remote, output_root=tmp_path)
        session.call_tool('reset_task')
        original = connections[0].writer
        class TrickleWriter:
            def write(self, data):
                for i in range(0, len(data), 50):
                    original.write(data[i:i+50])
                    original.flush()
                    time.sleep(.03)
                return len(data)
            def flush(self):
                original.flush()
        connections[0].writer = TrickleWriter()
        remote.timeout_s = .25
        result = session.call_tool('observe')
        assert result['structuredContent']['error'] == 'remote_timeout'
        assert session.status == 'aborted'
