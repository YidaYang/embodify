"""Backend proxy. Commands are never retried; recovery requires a new reset."""
from __future__ import annotations

import json
import math
import queue
import time
from collections import deque
from pathlib import Path

from .base import COORDINATED_STOP_REASONS, Backend, BackendDisconnected, Catalogue, CatalogueTexts, ExtraResult, McpToolError, MotionReport
from .coordinated import validate_arm_results
from .transport import TransportError, connect_tcp, start_process
from .wire import codecs, decode_image, pack_task, unpack_info, unpack_snapshot, unpack_task


class RemoteCatalogue(Catalogue):
    def __init__(self, backend):
        self.backend = backend
        self.selected = None
        self.payload = None

    def texts(self):
        return self.backend._texts

    def resolve(self, args, *, locked):
        reply = self.backend._call("catalogue.resolve", {"args": args, "locked": locked})
        return unpack_task(reply["task"]), int(reply["variant"])

    def commit(self, task, variant):
        request = {"task": pack_task(task), "variant": variant}
        reply = self.backend._call("catalogue.commit", request)
        self.selected = request
        self.payload = reply["task"]
        self.backend._idle = reply["idle"]

    def task_payload(self):
        # Available after a failed connection so abort logging needs no network access.
        if self.payload is None:
            task, variant = self.resolve({}, locked=False)
            self.commit(task, variant)
        return dict(self.payload)

    def idle_payload(self):
        return dict(self.backend._idle)

    def listing(self, args):
        return self.backend._call("catalogue.listing", {"args": args})


class RemoteBackend(Backend):
    def __init__(self, *, command=None, host=None, port=None, cwd=None,
                 timeout_s=120.0, connect_timeout_s=30.0, heartbeat_timeout_s=30.0,
                 close_timeout_s=10.0,
                 frame_stride=1):
        if (command is None) == (host is None):
            raise ValueError("choose exactly one backend command or TCP address")
        if command is not None and (not isinstance(command, (list, tuple)) or not command
                                   or any(not isinstance(v, str) or not v for v in command)):
            raise ValueError("command must be a non-empty argv list (no shell)")
        if host is not None and (not isinstance(host, str) or not host or type(port) is not int or not 1 <= port <= 65535):
            raise ValueError("invalid backend TCP address")
        if type(frame_stride) is not int or not 1 <= frame_stride <= 1000:
            raise ValueError("frame_stride must be 1..1000")
        for value in (timeout_s, connect_timeout_s, heartbeat_timeout_s, close_timeout_s):
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError("timeouts must be finite and positive")
        self.command, self.host, self.port, self.cwd = command, host, port, cwd
        self.timeout_s, self.connect_timeout_s = float(timeout_s), float(connect_timeout_s)
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.close_timeout_s = float(close_timeout_s)
        self.frame_stride = frame_stride
        self._connection = None
        self._ident = 0
        self._active = False
        self._judged = None
        self._last_step = 0
        self._info = None
        self._catalogue = RemoteCatalogue(self)
        self._events = deque(maxlen=256)
        self._open()

    @classmethod
    def from_config(cls, path, *, frame_stride=None):
        path = Path(path).resolve()
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        transport = config.pop("transport")
        allowed = {"command", "cwd", "host", "port", "timeout_s", "connect_timeout_s", "heartbeat_timeout_s", "close_timeout_s", "frame_stride"}
        if set(config) - allowed:
            raise ValueError("unknown remote configuration keys")
        if transport not in ("stdio", "tcp") or (transport == "stdio") != ("command" in config):
            raise ValueError("transport and endpoint disagree")
        if config.get("cwd"):
            config["cwd"] = str((path.parent / config["cwd"]).resolve())
        if frame_stride is not None:
            config["frame_stride"] = frame_stride
        return cls(**config)

    def _open(self):
        try:
            connection = (start_process(self.command, self.cwd) if self.command is not None else
                          connect_tcp(self.host, self.port, self.connect_timeout_s))
            self._connection = connection
            interval = min(1.0, self.heartbeat_timeout_s / 3)
            connection.keepalive(interval, self.heartbeat_timeout_s)
            reply = self._exchange("hello", {"codecs": codecs(), "frame_stride": self.frame_stride},
                                   timeout=self.connect_timeout_s)
            info = unpack_info(reply["info"])
            texts = CatalogueTexts(**reply["texts"])
            if self._info is not None and (info != self._info or texts != self._texts or reply["extra_tools"] != self._extra_tools):
                raise ValueError("backend capabilities changed; start a new MCP session")
            self._info, self._texts, self._idle = info, texts, reply["idle"]
            self._params, self._extra_tools, self._codec = reply["journal_params"], reply["extra_tools"], reply["codec"]
            if self._catalogue.selected is not None:
                restored = self._exchange("catalogue.commit", self._catalogue.selected)
                self._catalogue.payload, self._idle = restored["task"], restored["idle"]
        except BackendDisconnected:
            raise
        except Exception:
            self._disconnect()
            raise BackendDisconnected("remote_connect_failed", "Cannot connect to the backend; check the backend configuration and retry reset_task")

    def _disconnect(self):
        connection, self._connection = self._connection, None
        self._active, self._judged = False, None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    def _call(self, method, params=None, callback=None):
        if self._connection is None:
            self._open()
        return self._exchange(method, params or {}, callback=callback)

    def _exchange(self, method, params, *, callback=None, timeout=None):
        connection = self._connection
        self._ident += 1
        ident = self._ident
        started = time.monotonic()
        deadline = started + (self.timeout_s if timeout is None else timeout)
        received, sent = connection.bytes_received, connection.bytes_sent
        progress_count, server_s, error = 0, None, None
        previous_step = self._last_step
        try:
            connection.send({"kind": "request", "id": ident, "method": method, "params": params},
                            timeout=max(0.001, deadline - time.monotonic()))
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("backend operation deadline")
                if time.monotonic() - connection.last_received > self.heartbeat_timeout_s:
                    raise TransportError("backend heartbeat lost")
                try:
                    reply = connection.receive(min(remaining, 0.2))
                except queue.Empty:
                    continue
                if reply.get("id") != ident:
                    raise ValueError("backend response id mismatch")
                kind = reply.get("kind")
                if kind == "progress":
                    if method not in ("move_to", "set_gripper", "control_arms"):
                        raise ValueError("unexpected progress")
                    snapshot = self._snapshot(reply, full=False)
                    if snapshot.step != previous_step + 1:
                        raise ValueError("non-sequential progress")
                    previous_step = snapshot.step
                    progress_count += 1
                    if callback is not None:
                        callback(snapshot)
                    continue
                server_s = float(reply.get("server_s", 0.0))
                if kind == "error":
                    error = reply["code"]
                    if reply.get("fatal"):
                        raise BackendDisconnected(error, reply["message"])
                    raise McpToolError(error, reply["message"])
                if kind != "result":
                    raise ValueError("unexpected backend message")
                result = reply["result"]
                if method in ("reset", "observe", "move_to", "set_gripper", "control_arms"):
                    snapshot = self._snapshot(result)
                    if method in ("move_to", "set_gripper", "control_arms"):
                        if (type(result["executed_steps"]) is not int
                                or result["executed_steps"] != progress_count
                                or progress_count > params["max_steps"] or snapshot.step != previous_step):
                            raise ValueError("motion report disagrees with progress")
                        allowed = (COORDINATED_STOP_REASONS if method == "control_arms" else
                                   self.info.move_stop_reasons if method == "move_to" else self.info.gripper_stop_reasons)
                        if result["stop_reason"] not in {"limit"} | {r[0] for r in allowed}:
                            raise ValueError("invalid motion stop reason")
                        if method == "control_arms":
                            validate_arm_results(result["arm_results"], self.info.arm_names)
                        distance = float(result["remaining_distance_m"])
                        if not math.isfinite(distance) or distance < 0 or not isinstance(result["telemetry"], dict):
                            raise ValueError("invalid motion report")
                elif method == "call_extra":
                    # Decode before leaving the guarded exchange so corrupt media aborts the episode.
                    result = ExtraResult(result["data"], [decode_image(img) for img in result["images"]], result["log"])
                return result
        except BackendDisconnected as exc:
            error = exc.code
            self._disconnect()
            raise
        except McpToolError:
            raise
        except TimeoutError:
            error = "remote_timeout"
            self._disconnect()
            raise BackendDisconnected(error, "The backend call timed out and the outcome of the action is unknown; this episode was aborted, call reset_task again")
        except (TransportError, OSError):
            error = "remote_disconnected"
            self._disconnect()
            raise BackendDisconnected(error, "The backend connection was lost; this episode was aborted, call reset_task again")
        except Exception:
            error = "remote_protocol_error"
            self._disconnect()
            raise BackendDisconnected(error, "Handling the backend response or progress failed; this episode was aborted, call reset_task again")
        finally:
            self._events.append({"method": method, "roundtrip_s": time.monotonic() - started,
                                 "server_s": server_s, "received_bytes": connection.bytes_received - received,
                                 "sent_bytes": connection.bytes_sent - sent, "progress_steps": progress_count,
                                 "error": error})

    def _snapshot(self, value, full=True):
        snapshot = unpack_snapshot(value["snapshot"])
        if set(snapshot.arms) != set(self.info.arm_names):
            raise ValueError("backend arms changed")
        expected = {c.name for c in self.info.cameras}
        if not set(snapshot.images) <= expected or (full and set(snapshot.images) != expected):
            raise ValueError("backend cameras changed")
        judged = value["evaluation"]
        if judged is not None and type(judged) is not bool:
            raise ValueError("invalid evaluation value")
        self._judged = judged
        self._last_step = snapshot.step
        return snapshot

    def _require_episode(self):
        if not self._active:
            raise McpToolError("remote_reset_required", "The backend needs a new reset_task; the old episode cannot continue")

    @property
    def info(self):
        return self._info

    @property
    def catalogue(self):
        return self._catalogue

    def reset(self, task, variant):
        result = self._call("reset", {"task": pack_task(task), "variant": variant})
        snapshot = self._snapshot(result)
        if isinstance(result.get("task"), dict):
            # Some backends learn the instruction only when the episode starts.
            self._catalogue.payload = dict(result["task"])
        self._active = True
        return snapshot

    def observe(self):
        self._require_episode()
        return self._snapshot(self._call("observe"))

    def _motion(self, method, params, callback):
        self._require_episode()
        result = self._call(method, params, callback)
        return MotionReport(result["stop_reason"], int(result["executed_steps"]), self._snapshot(result),
                            float(result["remaining_distance_m"]), result["telemetry"], result.get("arm_results", {}))

    def move_to(self, arm, pos, quat, *, max_steps, on_step=None):
        return self._motion("move_to", {"arm": arm, "pos": pos.tolist(), "quat": quat.tolist(),
                                       "max_steps": max_steps}, on_step)

    def set_gripper(self, arm, close, *, max_steps, on_step=None):
        return self._motion("set_gripper", {"arm": arm, "close": close, "max_steps": max_steps}, on_step)

    def control_arms(self, targets, *, max_steps, on_step=None):
        if not self.info.supports_coordinated_control:
            return super().control_arms(targets, max_steps=max_steps, on_step=on_step)
        return self._motion("control_arms", {"targets": targets, "max_steps": max_steps}, on_step)

    def success(self):
        # Called from each progress callback: never issue a nested RPC here.
        if self._judged is None:
            raise RuntimeError("backend evaluation unavailable")
        return self._judged

    def episode_audit(self):
        self._require_episode()
        result = self._call("episode_audit", {"include_evaluation": True})
        if isinstance(result, dict) and result.get("final_evaluation") is True:
            judged = result["evaluation"]
            if judged is not None and type(judged) is not bool:
                raise ValueError("invalid final evaluation")
            self._judged = judged
            return result["audit"]
        return result  # older P2 server: audit only, keep its last evaluation

    def end_episode(self):
        if self._connection is not None:
            self._call("end_episode")
        self._active, self._judged = False, None

    def close(self):
        try:
            if self._connection is not None and not self._connection.closed.is_set():
                self._exchange("close", {}, timeout=min(self.timeout_s, self.close_timeout_s))
        except Exception:
            pass
        finally:
            self._disconnect()

    def journal_params(self):
        return dict(self._params, remote={"transport": "stdio" if self.command is not None else "tcp",
                                        "frame_stride": self.frame_stride, "codec": self._codec,
                                        "timeout_s": self.timeout_s, "connect_timeout_s": self.connect_timeout_s,
                                        "heartbeat_timeout_s": self.heartbeat_timeout_s,
                                        "close_timeout_s": self.close_timeout_s})

    def drain_transport_events(self):
        events = list(self._events)
        self._events.clear()
        return events

    def extra_tools(self):
        return self._extra_tools

    def call_extra(self, name, args, *, running):
        return self._call("call_extra", {"name": name, "args": args, "running": running})
