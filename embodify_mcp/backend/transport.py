"""Bounded newline JSON transport over subprocess pipes or TCP (Python 3.8)."""
from __future__ import annotations

import queue
import select
import socket
import subprocess
import threading
import time

from .wire import MAX_LINE, VERSION, dumps, loads


class TransportError(Exception):
    pass


class LineTransport:
    def __init__(self, reader, writer, shutdown):
        self.reader, self.writer, self.shutdown = reader, writer, shutdown
        self.messages = queue.Queue(maxsize=64)
        self.closed = threading.Event()
        self.peer_closed = threading.Event()
        self.write_lock = threading.Lock()
        self.close_lock = threading.Lock()
        self.last_received = time.monotonic()
        self.bytes_received = 0
        self.bytes_sent = 0
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _put(self, value):
        while not self.closed.is_set():
            try:
                self.messages.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def _read(self):
        # A large JSON image may take seconds to arrive. Bytes arriving are evidence
        # of a live peer, even before the terminating newline (heartbeats share this stream).
        pending = bytearray()
        try:
            while not self.closed.is_set():
                chunk = self.reader.read1(64 * 1024)
                if not chunk:
                    raise TransportError("backend stream closed")
                self.bytes_received += len(chunk)
                self.last_received = time.monotonic()
                pending.extend(chunk)
                while True:
                    end = pending.find(b"\n")
                    if end < 0:
                        if len(pending) >= MAX_LINE:
                            raise TransportError("invalid backend frame")
                        break
                    if end + 1 > MAX_LINE:
                        raise TransportError("invalid backend frame")
                    line = bytes(pending[:end + 1])
                    del pending[:end + 1]
                    value = loads(line)
                    if value.get("kind") != "heartbeat":
                        self._put(value)
        except Exception as exc:
            self.peer_closed.set()
            self._put(TransportError(str(exc)))

    def send(self, value, timeout=30.0, *, heartbeat=False):
        if self.closed.is_set():
            raise TransportError("connection closed")
        data = dumps(dict(value, v=VERSION))
        finished = threading.Event()
        errors = []

        def write():
            try:
                # A data write already queued on the link makes a separate heartbeat
                # unnecessary. Never let it time out merely waiting behind a large frame.
                if not self.write_lock.acquire(blocking=not heartbeat):
                    return
                try:
                    if self.closed.is_set():
                        raise TransportError("connection closed")
                    self.writer.write(data)
                    self.writer.flush()
                    self.bytes_sent += len(data)
                finally:
                    self.write_lock.release()
            except Exception as exc:
                errors.append(exc)
            finally:
                finished.set()
        threading.Thread(target=write, daemon=True).start()
        if not finished.wait(timeout):
            self.close()
            raise TimeoutError("backend write timeout")
        if errors:
            raise TransportError(str(errors[0]))

    def receive(self, timeout):
        try:
            item = self.messages.get(timeout=timeout)
        except queue.Empty:
            if self.closed.is_set():
                raise TransportError("connection closed")
            raise
        if isinstance(item, Exception):
            raise item
        return item

    def keepalive(self, interval=1.0, peer_timeout=30.0):
        def beat():
            while not self.closed.wait(interval):
                if time.monotonic() - self.last_received > peer_timeout:
                    self.close()
                    return
                try:
                    self.send({"kind": "heartbeat"}, timeout=peer_timeout, heartbeat=True)
                except Exception:
                    self.close()
                    return
        threading.Thread(target=beat, daemon=True).start()

    def close(self):
        with self.close_lock:
            if self.closed.is_set():
                return
            self.closed.set()
        try:
            self.shutdown()
        except OSError:
            pass


def socket_transport(sock):
    sock.settimeout(None)
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    stopped = threading.Event()

    class Reader:
        def read1(self, size):
            # Do not close a buffered makefile while another thread holds its read
            # lock: on Windows that can hang if the relay keeps its socket open.
            while not stopped.is_set():
                ready, _, _ = select.select([sock], [], [], 0.2)
                if ready and not stopped.is_set():
                    return sock.recv(size)
            return b""

    class Writer:
        def write(self, data):
            sock.sendall(data)
            return len(data)

        def flush(self):
            pass

    def shutdown():
        stopped.set()
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()
    return LineTransport(Reader(), Writer(), shutdown)


def connect_tcp(host, port, timeout):
    return socket_transport(socket.create_connection((host, port), timeout=timeout))


def start_process(command, cwd=None):
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=None, creationflags=flags)

    def shutdown():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1.0)
        proc.stdin.close()
        proc.stdout.close()
    return LineTransport(proc.stdout, proc.stdin, shutdown)
