"""Backend-only service. Physics stays on the calling thread; heartbeat does no physics."""
from __future__ import annotations

import json
import queue
import socket
import sys
import time
from dataclasses import asdict

from .base import McpToolError
from .transport import LineTransport, TransportError, socket_transport
from .wire import codecs, encode_image, pack_snapshot, pack_task, unpack_task, vector


def serve_connection(backend, connection, *, heartbeat_interval=1.0, peer_timeout=30.0):
    """One connection owns the backend until disconnect; every new owner starts without an episode."""
    connection.keepalive(heartbeat_interval, peer_timeout)
    codec, stride, ready, active = "rgb-zlib", 1, False, False
    budget = backend.info.episode_budget

    def evaluation():
        if not backend.info.has_success:
            return None
        try:
            return bool(backend.success())
        except Exception:
            return None

    def observation(snapshot, images=True):
        return {"snapshot": pack_snapshot(snapshot, codec, images), "evaluation": evaluation()}

    try:
        while not connection.closed.is_set():
            try:
                request = connection.receive(0.2)
            except queue.Empty:
                continue
            if request.get("kind") != "request" or type(request.get("id")) is not int:
                raise TransportError("invalid request envelope")
            ident = request["id"]
            method, params = request.get("method"), request.get("params", {})
            started = time.monotonic()
            shutdown = False
            try:
                if not isinstance(params, dict):
                    raise McpToolError("invalid_input", "Backend parameters must be an object")
                if method != "hello" and not ready:
                    raise McpToolError("invalid_input", "Negotiate the backend protocol first")
                if method == "hello":
                    if ready:
                        raise McpToolError("invalid_input", "The backend protocol is already negotiated")
                    offers = params.get("codecs", [])
                    common = [c for c in codecs() if c in offers]
                    stride = params.get("frame_stride", 1)
                    if not common or type(stride) is not int or not 1 <= stride <= 1000:
                        raise McpToolError("invalid_input", "Unsupported image format or frame stride")
                    codec, ready = common[0], True
                    public_info = asdict(backend.info)
                    if public_info.get("gripper_opening_measured") is True:
                        public_info.pop("gripper_opening_measured")
                    if not public_info["supports_coordinated_control"]:
                        public_info.pop("supports_coordinated_control")
                    result = {"info": public_info, "texts": asdict(backend.catalogue.texts()),
                              "idle": backend.catalogue.idle_payload(), "codec": codec,
                              "journal_params": backend.journal_params(), "extra_tools": backend.extra_tools()}
                elif method == "ping":
                    result = {}
                elif method == "catalogue.resolve":
                    task, variant = backend.catalogue.resolve(params["args"], locked=params["locked"])
                    result = {"task": pack_task(task), "variant": variant}
                elif method == "catalogue.commit":
                    backend.catalogue.commit(unpack_task(params["task"]), params["variant"])
                    result = {"task": backend.catalogue.task_payload(), "idle": backend.catalogue.idle_payload()}
                elif method == "catalogue.listing":
                    result = backend.catalogue.listing(params["args"])
                elif method == "reset":
                    if active:
                        raise McpToolError("already_running", "End the current episode first")
                    task = unpack_task(params["task"])
                    snap = backend.reset(task, params["variant"])
                    active = True
                    budget = backend.info.episode_budget if task.episode_budget is None else task.episode_budget
                    # The instruction may only be known once the episode exists.
                    result = dict(observation(snap), task=backend.catalogue.task_payload())
                elif method in ("observe", "move_to", "set_gripper", "control_arms", "episode_audit"):
                    if not active:
                        raise McpToolError("remote_reset_required", "The backend has no active episode; call reset_task")
                    if method == "observe":
                        result = observation(backend.observe())
                    elif method == "episode_audit":
                        audit = backend.episode_audit()
                        result = ({"audit": audit, "evaluation": evaluation(), "final_evaluation": True}
                                  if params.get("include_evaluation") is True else audit)
                    else:
                        steps = params["max_steps"]
                        if type(steps) is not int or not 0 <= steps <= budget:
                            raise McpToolError("invalid_input", "Invalid action budget")
                        # Buffer one step so the final callback always includes fresh images,
                        # even when frame_stride skips intermediate images. Evaluation is captured
                        # immediately at that step, never queried later for a buffered snapshot.
                        pending = [None]
                        count = [0]

                        def flush(final=False):
                            if pending[0] is not None:
                                snap, judged, ordinal = pending[0]
                                connection.send({"kind": "progress", "id": ident,
                                                 "snapshot": pack_snapshot(snap, codec,
                                                    final or ordinal == 1 or ordinal % stride == 0),
                                                 "evaluation": judged}, timeout=peer_timeout)
                                pending[0] = None

                        def progress(snap):
                            if connection.closed.is_set() or connection.peer_closed.is_set():
                                raise TransportError("peer disconnected")
                            flush()
                            count[0] += 1
                            pending[0] = (snap, evaluation(), count[0])

                        try:
                            if method == "control_arms":
                                report = backend.control_arms(params["targets"], max_steps=steps, on_step=progress)
                            elif method == "move_to":
                                report = backend.move_to(params["arm"], vector(params["pos"], 3),
                                    vector(params["quat"], 4), max_steps=steps, on_step=progress)
                            else:
                                if type(params["close"]) is not bool:
                                    raise McpToolError("invalid_input", "close must be a boolean")
                                report = backend.set_gripper(params["arm"], params["close"],
                                    max_steps=steps, on_step=progress)
                        finally:
                            flush(final=True)
                        result = {"stop_reason": report.stop_reason, "executed_steps": report.executed_steps,
                                  "remaining_distance_m": report.remaining_distance_m,
                                  "telemetry": report.telemetry, **observation(report.snapshot)}
                        if method == "control_arms":
                            result["arm_results"] = report.arm_results
                elif method == "call_extra":
                    if params["running"] != active:
                        raise McpToolError("remote_reset_required", "Inconsistent session state; call reset_task again")
                    extra = backend.call_extra(params["name"], params["args"], running=active)
                    result = {"data": extra.data, "log": extra.log,
                              "images": [encode_image(img, codec) for img in extra.images]}
                elif method in ("end_episode", "close"):
                    backend.end_episode()
                    active = False
                    result = {}
                    shutdown = method == "close"
                else:
                    raise McpToolError("unknown_method", "Unknown backend method")
                response = {"kind": "result", "id": ident, "result": result}
            except McpToolError as exc:
                response = {"kind": "error", "id": ident, "code": exc.code, "message": exc.message, "fatal": False}
            except (TransportError, BrokenPipeError, ConnectionError):
                raise
            except Exception as exc:
                # Do not send arbitrary backend exception text (paths / privileged data) to the Agent.
                print("[backend] {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
                response = {"kind": "error", "id": ident, "code": "remote_backend_error",
                            "message": "Backend execution failed; this episode was aborted, call reset_task again", "fatal": True}
                shutdown = True
            response["server_s"] = time.monotonic() - started
            connection.send(response, timeout=peer_timeout)
            if shutdown:
                break
    except (TransportError, OSError, TimeoutError, ValueError) as exc:
        print("[backend] connection ended: {}".format(exc), file=sys.stderr)
    finally:
        connection.close()
        # Same thread that created/stepped the renderer must release it.
        backend.end_episode()


def main(argv=None):
    from ..mcp import build_backend, build_parser, split_protocol_stdout

    parser = build_parser()
    parser.description = "Embodify backend service (runs no MCP and keeps no local Agent logs)"
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=None, help="Omit to use stdio; 0 lets the system choose a port")
    parser.add_argument("--peer-timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.backend == "remote":
        parser.error("The backend service cannot wrap the remote backend")
    if args.peer_timeout < 2:
        parser.error("--peer-timeout must be at least 2 seconds")
    sink = split_protocol_stdout()
    backend = build_backend(args)
    try:
        if args.listen_port is None:
            connection = LineTransport(sys.stdin.buffer, sink.buffer, lambda: None)
            serve_connection(backend, connection, peer_timeout=args.peer_timeout)
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((args.listen_host, args.listen_port))
                listener.listen(1)
                print("BACKEND_READY " + json.dumps({"host": args.listen_host, "port": listener.getsockname()[1]}),
                      file=sys.stderr, flush=True)
                while True:
                    client, _ = listener.accept()
                    # Single owner, sequential connections; no sharing physics between clients.
                    serve_connection(backend, socket_transport(client), peer_timeout=args.peer_timeout)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
