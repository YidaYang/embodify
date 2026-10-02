"""Tools-only stdio MCP server with backend-owned control loops.

The host owns the model, conversation, memory and skills. This process handles
sessions, bounded actions, public observations and private operator journals.
Heavy simulator imports are deferred to the selected backend.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import __version__
from .backend.base import LIMIT, ArmState, Backend, BackendDisconnected, McpToolError, Snapshot
from .backend.coordinated import validate_arm_results
from .backend.stepped import DEFAULT_MAX_STEPS_PER_CALL
from .imaging import encode_png
from .journal import Journal, RunManifest
from .mcp_tools import build_tools
from .primitives.controller import target_quat_from_rpy_delta

#: Run logs live in the user's home rather than in whatever folder the agent was started from.
DEFAULT_OUTPUT_ROOT = "~/.embodify/runs"


class McpSession:
    """One MCP process, one backend, at most one running episode at a time."""

    def __init__(
        self,
        backend: Backend,
        *,
        output_root: Path = Path("out/mcp"),
        max_steps_per_call: int = DEFAULT_MAX_STEPS_PER_CALL,
        task_locked: bool = False,
    ) -> None:
        self.backend = backend
        self.info = backend.info
        self.catalogue = backend.catalogue
        self.task_locked = bool(task_locked)
        self.output_root = Path(output_root)
        self.max_steps_per_call = int(max_steps_per_call)
        if self.max_steps_per_call <= 0:
            raise ValueError("max_steps_per_call must be positive")
        self.status = "idle"
        self._episode_open = False
        self.run_id: Optional[str] = None
        self.journal: Optional[Journal] = None
        self.step_index = 0
        # Per episode: a task may set its own limit (TaskSpec.episode_budget).
        self.episode_budget = int(self.info.episode_budget)
        self._run_counter = 0
        self._last_error: Optional[str] = None
        # Ground truth, kept strictly on the journal side of the wall. See `_note_success`.
        self._success_at: Optional[int] = None
        self._success_probe_failed = False
        self._last_obs: Optional[Snapshot] = None
        self._pending_grippers: Set[str] = set()
        # Which server launch an episode came from. Every reset gets its own timestamp folder, so
        # without this the monitor could not tell "r2 of this launch" from "r2 of yesterday's".
        self.session_id = time.strftime("%Y%m%d-%H%M%S")

    # -- public tool surface ---------------------------------------------

    def tools(self) -> List[Dict[str, Any]]:
        """The tools exposed to the host Agent."""
        return build_tools(self.info, self.catalogue.texts()) + list(self.backend.extra_tools())

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        args = arguments if arguments is not None else {}
        was_running = self.status == "running"
        previous_journal = self.journal
        self.backend.drain_transport_events()
        step_from = self.step_index
        started = time.time()
        result = self._dispatch(name, args)
        # Only calls that touch an episode are recorded, and they go to that episode's journal.
        # A `list_tasks` between two episodes belongs to neither, and appending it to the one
        # that already finished would make that run look longer than it was.
        if self.journal is not None and (was_running or self.status == "running" or self.journal is not previous_journal):
            self._record_tool_call(name, args if isinstance(args, dict) else {"_raw": args}, result, step_from, started)
        return result

    def _dispatch(self, name: str, args: Any) -> Dict[str, Any]:
        try:
            if not isinstance(args, dict):
                raise McpToolError("invalid_input", "arguments must be a JSON object")
            if name == "reset_task":
                return self.reset_task(args)
            if name == "observe":
                return self.observe()
            if name == "move_relative":
                return self.move_relative(args)
            if name == "control_arms" and self.info.supports_coordinated_control:
                return self.control_arms(args)
            if name == "set_gripper":
                return self.set_gripper(args)
            if name == "stop_episode":
                return self.stop_episode()
            if name == "get_session_info":
                return self.get_session_info()
            if name == "list_tasks":
                return self.list_tasks(args)
            if name in {tool["name"] for tool in self.backend.extra_tools()}:
                return self._call_extra(name, args)
            raise McpToolError("unknown_tool", f"Unknown tool: {name}")
        except BackendDisconnected as exc:
            self._last_error = f"{exc.code}: {exc.message}"
            self._abort_episode(exc.code)
            return self._error_result(exc.code, exc.message)
        except McpToolError as exc:
            self._last_error = f"{exc.code}: {exc.message}"
            return self._error_result(exc.code, exc.message)
        except Exception as exc:  # pragma: no cover - real MuJoCo errors are environment-specific
            self._last_error = f"sim_error: {type(exc).__name__}: {exc}"
            print("[embodify-mcp] " + self._last_error, file=sys.stderr)
            return self._error_result("sim_error", "Backend execution failed; inspect operator logs before continuing")

    # -- tool implementations -------------------------------------------

    def reset_task(self, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.status == "running":
            raise McpToolError("already_running", "An episode is already running; call stop_episode first")
        task, variant = self.catalogue.resolve(args or {}, locked=self.task_locked)
        self.catalogue.commit(task, variant)
        self.episode_budget = int(self.info.episode_budget if task.episode_budget is None else task.episode_budget)
        self._run_counter += 1
        self.run_id = f"r{self._run_counter}"
        # The public ID is short; the parent folder prevents collisions across server launches.
        run_root = self.output_root / time.strftime("%Y%m%d-%H%M%S")
        manifest = RunManifest(
            run_id=self.run_id,
            arm="mcp-agent",
            rung="mcp",
            split_digest="",
            frozen_digest=None,
            privileged=False,
            sim=self.info.name,
            notes="MCP session; the Agent gets no success or privileged state",
        )
        self.journal = Journal(run_root, manifest)
        self._episode_open = True
        self.step_index = 0
        self._last_obs = None
        # Every run records which controller constants it used and whether they were ever probed,
        # so an unverified run can never be mistaken for a verified one after the fact.
        self.journal.event(
            "controller_params",
            max_steps_per_call=self.max_steps_per_call,
            max_steps=self.episode_budget,
            **self.backend.journal_params(),
        )
        # Which scene this run actually used. Without it, a run where the Agent wandered off to a
        # different task is indistinguishable afterwards from one that stayed put. The camera and
        # arm lists tell the monitor what to draw, since they differ between backends.
        self.journal.event(
            "scene",
            **self.catalogue.task_payload(),
            chosen_by="server" if self.task_locked else "agent_or_default",
            session=self.session_id,
            backend=self.info.name,
            cameras=[{"name": c.name, "label": c.label} for c in self.info.cameras],
            arms=list(self.info.arm_names),
        )
        self._last_error = None
        # Must be cleared before the first observation of the new episode is recorded,
        # or a latched success from the previous run would be attributed to this one.
        self._success_at = None
        self._success_probe_failed = False
        obs = self._sim_call(self.backend.reset, task, variant)
        self._pending_grippers.clear()
        self.step_index = int(obs.step)
        self._last_obs = obs
        self.status = "running"
        self._record_observation(obs, "reset")
        data = {
            "run": self.run_id,
            "task": self.catalogue.task_payload(),
            "images": self._image_descriptors(),
            "state": self._state_payload(obs),
            "step": self.step_index,
            "steps_left": self._steps_left(),
        }
        return self._result(data, obs)

    def observe(self) -> Dict[str, Any]:
        self._require_running()
        obs = self._sim_call(self.backend.observe)
        self.step_index = int(obs.step)
        self._last_obs = obs
        self._record_observation(obs, f"observe_{self.step_index:04d}")
        return self._result(
            {
                "images": self._image_descriptors(),
                "state": self._state_payload(obs),
                "step": self.step_index,
                "steps_left": self._steps_left(),
                "status": self.status,
            },
            obs,
        )

    def move_relative(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self._require_running()
        arm = self._arm_arg(args)
        delta = _vector3(args.get("delta_xyz"), "delta_xyz must be 3 finite numbers, in meters")
        delta_rpy = _vector3(args.get("delta_rpy", [0.0, 0.0, 0.0]), "delta_rpy must be 3 finite numbers, in radians")
        current = self._sim_call(self.backend.observe)
        allowed, capped_by = self._claim_steps(current)
        start = current.arms[arm]
        start_pos = np.asarray(start.pos, dtype=np.float64).reshape(3).copy()
        target = start_pos + delta
        target_quat = target_quat_from_rpy_delta(start.quat, delta_rpy)
        report = self._sim_call(
            self.backend.move_to, arm, target, target_quat, max_steps=allowed, on_step=self._on_step("move")
        )
        obs = report.snapshot
        self.step_index = int(obs.step)
        self._last_obs = obs
        stop_reason = capped_by if report.stop_reason == LIMIT else report.stop_reason
        achieved = np.asarray(obs.arms[arm].pos, dtype=np.float64).reshape(3) - start_pos
        action: Dict[str, Any] = {"arm": arm} if self._multi_arm else {}
        action.update(
            {
                "delta_xyz": [float(x) for x in delta],
                "delta_rpy": [float(x) for x in delta_rpy],
                "executed_steps": int(report.executed_steps),
                "stop_reason": stop_reason,
                "achieved_delta_xyz": [float(x) for x in achieved],
                "remaining_distance_m": float(report.remaining_distance_m),
            }
        )
        data = {
            "images": self._image_descriptors(),
            "state": self._state_payload(obs),
            "action": action,
            "step": self.step_index,
            "steps_left": self._steps_left(),
            "status": self.status,
        }
        if self.journal is not None:
            self.journal.event(
                "mcp_move",
                id=f"a{self.step_index}",
                **action,
                **report.telemetry,
            )
        return self._result(data, obs)

    def control_arms(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self._require_running()
        if set(args) != {"arms"} or not isinstance(args["arms"], dict):
            raise McpToolError("invalid_input", "Provide an arms object and no other top-level arguments")
        requested = args["arms"]
        if not requested or set(requested) - set(self.info.arm_names):
            raise McpToolError("invalid_input", "arms must be nonempty and use valid arm names")
        parsed = {}
        for name, value in requested.items():
            if (not isinstance(value, dict) or not value
                    or set(value) - {"delta_xyz", "delta_rpy", "gripper"}):
                raise McpToolError("invalid_input", "Each arm takes only pose increments and an optional gripper target, and must not be empty")
            deltas = []
            for field in ("delta_xyz", "delta_rpy"):
                raw = value.get(field, [0., 0., 0.])
                if (not isinstance(raw, list) or len(raw) != 3
                        or any(type(x) not in (int, float) for x in raw)):
                    raise McpToolError("invalid_input", "Pose increments must be three finite numbers")
                deltas.append(_vector3(raw, "Pose increments must be three finite numbers"))
            if "gripper" in value and value["gripper"] not in ("open", "close"):
                raise McpToolError("invalid_input", "gripper must be open or close")
            parsed[name] = deltas
        # A single snapshot is the origin for BOTH relative targets and held arms.
        current = self._sim_call(self.backend.observe)
        allowed, capped_by = self._claim_steps(current)
        targets = {}
        for name, start in current.arms.items():
            delta, rotation = parsed.get(name, (np.zeros(3), np.zeros(3)))
            targets[name] = {"pos": start.pos + delta,
                             "quat": target_quat_from_rpy_delta(start.quat, rotation)}
            if "gripper" in requested.get(name, {}):
                targets[name]["gripper"] = requested[name]["gripper"]
        report = self._sim_call(self.backend.control_arms, targets, max_steps=allowed,
                                on_step=self._on_step("control_arms"))
        feedback = validate_arm_results(report.arm_results, self.info.arm_names)
        obs = report.snapshot
        self.step_index, self._last_obs = int(obs.step), obs
        arms = {}
        for name in self.info.arm_names:
            arms[name] = dict(feedback[name], requested=name in requested,
                achieved_delta_xyz=(obs.arms[name].pos - current.arms[name].pos).tolist(),
                remaining_delta_xyz=(targets[name]["pos"] - obs.arms[name].pos).tolist())
            if feedback[name]["gripper_command_applied"]:
                self._pending_grippers.discard(name)
        action = {"arms": arms, "executed_steps": report.executed_steps,
                  "stop_reason": capped_by if report.stop_reason == LIMIT else report.stop_reason}
        if self.journal is not None:
            self.journal.event("mcp_control_arms", requested=requested, **action, **report.telemetry)
            if not report.executed_steps:
                self._record_observation(obs, f"control_arms_noop_{self.step_index:04d}")
        return self._result({"images": self._image_descriptors(), "state": self._state_payload(obs),
                             "action": action, "step": self.step_index,
                             "steps_left": self._steps_left(), "status": self.status}, obs)

    def set_gripper(self, args: Dict[str, Any]) -> Dict[str, Any]:
        self._require_running()
        arm = self._arm_arg(args)
        state = args.get("state")
        if not isinstance(state, str) or state not in ("open", "close"):
            raise McpToolError("invalid_input", "state must be open or close")
        current = self._sim_call(self.backend.observe)
        obs = current
        opening_before = _optional_float(current.arms[arm].gripper_opening)
        widths = [opening_before]
        steps = 0
        if state != current.arms[arm].gripper_command or arm in self._pending_grippers:
            # The budget is claimed *before* the backend flips its latch. Flipping first and then
            # executing zero steps would report "close" while the fingers are still wide open —
            # a state the Agent has no way to detect.
            allowed, capped_by = self._claim_steps(current)
            # A matching target is not enough for no_op after a capped or failed call.
            self._pending_grippers.add(arm)
            report = self._sim_call(
                self.backend.set_gripper, arm, state == "close", max_steps=allowed, on_step=self._on_step("gripper")
            )
            obs = report.snapshot
            steps = int(report.executed_steps)
            stop_reason = capped_by if report.stop_reason == LIMIT else report.stop_reason
            if report.stop_reason != LIMIT:
                self._pending_grippers.discard(arm)
            widths = list(report.telemetry.get("openings_m") or [opening_before, _optional_float(obs.arms[arm].gripper_opening)])
            self.step_index = int(obs.step)
            self._last_obs = obs
        else:
            stop_reason = "no_op"
            self.step_index = int(obs.step)
            self._last_obs = obs
            self._record_observation(obs, f"gripper_noop_{self.step_index:04d}")
        opening_after = _optional_float(obs.arms[arm].gripper_opening)
        action: Dict[str, Any] = {"arm": arm} if self._multi_arm else {}
        action.update(
            {
                "gripper": state,
                "executed_steps": steps,
                "stop_reason": stop_reason,
                "opening_before_m": opening_before,
                "opening_after_m": opening_after,
            }
        )
        data = {
            "images": self._image_descriptors(),
            "state": self._state_payload(obs),
            "action": action,
            "step": self.step_index,
            "steps_left": self._steps_left(),
            "status": self.status,
        }
        if self.journal is not None:
            record = dict(action)
            record["state"] = record.pop("gripper")
            self.journal.event("mcp_gripper", id=f"g{self.step_index}", openings_m=widths, **record)
        return self._result(data, obs)

    def _abort_episode(self, reason: str) -> None:
        if not self._episode_open or self.journal is None:
            return
        self.status = "aborted"
        self._episode_open = False
        self._pending_grippers.clear()
        self.journal.event("abort", reason=reason, step=self.step_index)
        self.journal.finish({
            "run": self.run_id, "status": "aborted", "reason": reason,
            "steps": self.step_index, "task": self.catalogue.task_payload(),
            "task_locked": self.task_locked, "backend": self.info.name,
            "success": True if self._success_at is not None else None,
            "success_at_step": self._success_at, "success_at_stop": None, "tool": "mcp",
        })

    def close(self) -> None:
        try:
            self._abort_episode("host_disconnected")
        finally:
            self.backend.close()

    def stop_episode(self) -> Dict[str, Any]:
        if self.status != "running":
            return self._error_result("not_running", "No episode is running")
        assert self.journal is not None
        # The evaluation record, taken while the backend is still alive — `end_episode()` below
        # makes it unaskable forever. It goes to disk and nowhere else: the response built at the
        # end of this method must stay free of it, or the Agent could grade itself.
        #   ever  = whether the task was ever solved (latched along the way by `_note_success`)
        #   still = whether it still holds at the end. LIBERO re-checks its BDDL predicates every
        #           step and a placed bowl can be knocked off again, so the two may disagree; which
        #           one a success rate uses is the evaluator's choice, not made here.
        # When the backend cannot answer (or cannot judge at all, like a real robot) both are None:
        # unknown and failed are different things and must not be mixed.
        audit = self._sim_call(self.backend.episode_audit)
        # Some benchmarks only perform their final checks when finalizing.
        self._note_success(self.step_index)
        ever = self._success_at is not None
        still = self._query_success()
        unknown = not self.info.has_success or (self._success_probe_failed and not ever)
        summary = {
            "run": self.run_id,
            "status": "stopped",
            "steps": self.step_index,
            "task": self.catalogue.task_payload(),
            "task_locked": self.task_locked,
            "backend": self.info.name,
            "success": None if unknown else ever,
            "success_at_step": self._success_at,
            "success_at_stop": still,
            "tool": "mcp",
        }
        if audit is not None:
            summary["physics_audit"] = audit
        self._sim_call(self.backend.end_episode)
        summary_path = self.journal.finish(summary)
        events_path = self.journal.dir / "events.jsonl"
        frames_path = self.journal.dir / "images"
        self.status = "stopped"
        self._episode_open = False
        return {
            "content": [{"type": "text", "text": json.dumps({"run": self.run_id, "status": self.status}, ensure_ascii=False)}],
            "structuredContent": {
                "ok": True,
                "run": self.run_id,
                "status": self.status,
                "artifacts": {
                    "events": str(events_path),
                    "frames": str(frames_path),
                    "summary": str(summary_path),
                },
            },
        }

    def get_session_info(self) -> Dict[str, Any]:
        if self.status == "idle" or self.run_id is None:
            # Which scene a plain reset lands on has to be readable *before* the first reset,
            # otherwise picking a scene would mean guessing.
            idle: Dict[str, Any] = {"ok": True, "status": "idle"}
            idle.update(self.catalogue.idle_payload())
            idle["task_locked"] = self.task_locked
            return {
                "content": [{"type": "text", "text": json.dumps(idle, ensure_ascii=False)}],
                "structuredContent": idle,
            }
        obs = self._sim_call(self.backend.observe) if self.status == "running" else self._last_obs
        if obs is not None:
            self.step_index = int(obs.step)
            self._last_obs = obs
        payload: Dict[str, Any] = {
            "ok": True,
            "run": self.run_id,
            "status": self.status,
            "step": self.step_index,
            "max_steps": self.episode_budget,
            "steps_left": self._steps_left(),
            "task": self.catalogue.task_payload(),
            "task_locked": self.task_locked,
        }
        if obs is not None:
            payload["state"] = self._state_payload(obs)
        return {
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
            "structuredContent": payload,
        }

    def list_tasks(self, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """The benchmark catalogue.

        Deliberately *not* part of `get_session_info`: which tasks exist is a property of the
        benchmark, not of this run, and a tool named "session info" is the last place an Agent
        would look for it. Keeping them apart also keeps the layering honest — the catalogue is
        a constant for the whole process, the session payload changes every step.
        """
        payload = self.catalogue.listing(args or {})
        return {
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
            "structuredContent": payload,
        }

    # -- serialization ----------------------------------------------------

    def _call_extra(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Forward a call to a tool the backend provides.

        Such tools (fitting an attachment, for example) are often called between episodes, when
        there is no journal, so they get a session-level log ordered by time and the surrounding runs.
        """
        result = self._sim_call(self.backend.call_extra, name, args, running=self.status == "running")
        entry = {
            "time": time.time(),
            "session": self.session_id,
            "tool": name,
            "arguments": args,
            "status": self.status,
            "last_run": self.run_id,
            "result": result.data,
            **result.log,
        }
        self.output_root.mkdir(parents=True, exist_ok=True)
        with (self.output_root / "extra_tools.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": json.dumps(result.data, ensure_ascii=False, separators=(",", ":"))}
        ]
        for image in result.images:
            content.append(
                {"type": "image", "data": base64.b64encode(encode_png(image)).decode("ascii"), "mimeType": "image/png"}
            )
        return {"content": content, "structuredContent": {"ok": True, **result.data}}

    @property
    def _multi_arm(self) -> bool:
        return len(self.info.arms) > 1

    def _arm_arg(self, args: Dict[str, Any]) -> str:
        """Which arm this call acts on. Single-arm backends neither need nor accept an arm argument."""
        names = self.info.arm_names
        if len(names) == 1:
            return names[0]
        arm = args.get("arm")
        if arm is None:
            raise McpToolError(
                "invalid_input",
                f"This backend has {len(names)} arms ({', '.join(names)}); use arm to choose one.",
            )
        if not isinstance(arm, str) or arm not in names:
            raise McpToolError("invalid_input", f"No arm named {arm!r}. Options: {', '.join(names)}.")
        return arm

    def _state_payload(self, obs: Snapshot) -> Dict[str, Any]:
        """Single arm: flat (eef_pos_m / eef_rpy_rad / gripper). Several arms: {"arms": {arm name: the same three}}."""
        if not self._multi_arm:
            return self._arm_payload(obs.arms[self.info.arm_names[0]])
        return {"arms": {name: self._arm_payload(obs.arms[name]) for name in self.info.arm_names}}

    @staticmethod
    def _arm_payload(arm: ArmState) -> Dict[str, Any]:
        roll, pitch, yaw = _quat_xyzw_to_rpy(np.asarray(arm.quat, dtype=np.float64))
        return {
            "eef_pos_m": [float(x) for x in np.asarray(arm.pos, dtype=np.float64).reshape(3)],
            "eef_rpy_rad": {"roll_x": roll, "pitch_y": pitch, "yaw_z": yaw},
            "gripper": {
                "command": arm.gripper_command,
                "opening_m": _optional_float(arm.gripper_opening),
            },
        }

    def _image_descriptors(self) -> List[Dict[str, str]]:
        return [{"camera": camera.name, "mimeType": "image/png"} for camera in self.info.cameras]

    def _result(self, data: Dict[str, Any], obs: Snapshot) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": json.dumps(data, ensure_ascii=False, separators=(",", ":"))}
        ]
        for camera in self.info.cameras:
            image = obs.images.get(camera.name)
            if image is None:
                raise McpToolError("missing_camera", f"The observation has no image from camera {camera.name}")
            content.append(
                {
                    "type": "image",
                    "data": base64.b64encode(encode_png(image)).decode("ascii"),
                    "mimeType": "image/png",
                }
            )
        return {"content": content, "structuredContent": {"ok": True, **data}}

    def _query_success(self) -> Optional[bool]:
        """Ask the backend whether the task is solved. The answer never leaves this class.

        Wrapped rather than called directly because a backend that cannot answer must not take
        the episode down with it: the Agent is mid-task and success is not its business anyway.
        The failure is recorded once and the probe then goes quiet, so a broken `success()` does
        not write one line per step into the journal. A backend that declares it cannot judge
        success at all (a real robot) is never asked.
        """
        if self._success_probe_failed or not self.info.has_success:
            return None
        try:
            return bool(self.backend.success())
        except Exception as exc:  # noqa: BLE001 — see docstring
            self._success_probe_failed = True
            if self.journal is not None:
                self.journal.event("success_unavailable", error=repr(exc))
            return None

    def _note_success(self, step: int) -> None:
        """Latch success into the journal the first time it holds.

        Success is the one fact the Agent must never see — the whole point of this server is that
        it works from images alone. It is also the only number the evaluation needs, and
        `stop_episode` ends the episode, after which nobody can ask again. So it is polled here,
        written to the journal, and kept out of every tool response.

        Latched rather than sampled once at the end because LIBERO re-checks its BDDL predicates
        every step: a bowl placed correctly and then nudged off the plate would read as a failure
        at `stop_episode` even though the task was solved. Recording the step it first held also
        makes the frame directory searchable: `move_0137_agentview.webp` is the moment.
        """
        if self.journal is None or self._success_at is not None:
            return
        if self._query_success():
            self._success_at = int(step)
            self.journal.event("success", step=int(step))

    def _record_tool_call(
        self, name: str, args: Dict[str, Any], result: Dict[str, Any], step_from: int, started: float
    ) -> None:
        """Write down one tool call exactly as the Agent saw it, minus the pictures.

        `mcp_move` / `mcp_gripper` / `mcp_control_arms` are calibration records for motion
        tools. This is the other half: *what the Agent did*, in order, including the calls that
        failed and the ones that never moved the arm. The frames recorded since the previous
        call are the ones this call produced, which is what lets a replay line them up.

        The images are left out because they are already on disk as frames. The benchmark
        catalogue is left out because it is a constant of the process, ~27 KB of it, and a
        replay only needs to know that the Agent looked.
        """
        assert self.journal is not None
        data = dict(result.get("structuredContent") or {})
        for bulky in ("suites", "tasks"):
            if isinstance(data.get(bulky), list):
                data[bulky] = f"<{len(data[bulky])} items, omitted>"
        self.journal.event(
            "tool_call",
            name=name,
            arguments=args,
            ok=not result.get("isError", False),
            error=data.get("error"),
            step_from=int(step_from),
            step_to=int(self.step_index),
            started_at=started,
            duration_s=time.time() - started,
            transport=self.backend.drain_transport_events(),
            result=data,
        )

    def _on_step(self, prefix: str) -> Callable[[Snapshot], None]:
        """Record a frame and poll success on every step of an action: the monitor's live view and the success moment rely on it."""

        def record(obs: Snapshot) -> None:
            self.step_index = int(obs.step)
            self._last_obs = obs
            self._record_observation(obs, f"{prefix}_{self.step_index:04d}")

        return record

    def _record_observation(self, obs: Snapshot, label: str) -> None:
        if self.journal is None:
            return
        self.journal.event("observation", label=label, step=int(obs.step), state=self._state_payload(obs), image_cameras=list(obs.images))
        # Every stepped observation passes through here, so this is the one place that sees the
        # whole episode — cheaper than sprinkling the probe across the action loops.
        self._note_success(int(obs.step))
        for camera in self.info.cameras:
            image = obs.images.get(camera.name)
            if image is not None:
                self.journal.save_frame(f"{label}_{camera.name}", image)

    def _require_running(self) -> None:
        if self.status != "running":
            raise McpToolError("not_running", "Call reset_task first")

    def _steps_left(self) -> int:
        return max(0, self.episode_budget - int(self.step_index))

    def _claim_steps(self, obs: Snapshot) -> Tuple[int, str]:
        """How many internal steps this call may run, and which ceiling would end it.

        Returns the step allowance plus the `stop_reason` to report if the backend uses all of
        it without converging: "budget" when the episode ran out, "step_cap" when this single call
        did. Keeping the two apart is the whole point of having two budgets — "it ran out of
        simulator time" and "it is still moving, call again" are different situations.

        An exhausted budget used to return ok=true after executing nothing, so the Agent saw a
        successful call and an unchanged picture; its only sensible reading was "the controller is
        too weak", and it would retry forever. The episode budget is a fact the Agent has to know.
        """
        budget = self.episode_budget
        remaining = max(0, budget - int(obs.step))
        if remaining <= 0:
            raise McpToolError(
                "step_limit",
                f"The episode's step budget is used up ({int(obs.step)}/{budget} steps), "
                "so no more actions can run. Call stop_episode to end this run.",
            )
        if remaining <= self.max_steps_per_call:
            return remaining, "budget"
        return self.max_steps_per_call, "step_cap"

    @staticmethod
    def _error_result(code: str, message: str) -> Dict[str, Any]:
        return {
            "isError": True,
            "content": [{"type": "text", "text": json.dumps({"ok": False, "error": code, "message": message}, ensure_ascii=False)}],
            "structuredContent": {"ok": False, "error": code, "message": message},
        }

    @staticmethod
    def _sim_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
        # robosuite may print startup warnings; keep stdout clean for JSON-RPC.
        with contextlib.redirect_stdout(sys.stderr):
            return fn(*args, **kwargs)


def _vector3(value: Any, message: str) -> np.ndarray:
    """A triple from the Agent. A wrong type (a string, say) is invalid_input, just like a wrong length."""
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        raise McpToolError("invalid_input", message)
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        raise McpToolError("invalid_input", message)
    return arr


def _quat_xyzw_to_rpy(quat: np.ndarray) -> Tuple[float, float, float]:
    """Convert an xyzw quaternion to XYZ roll/pitch/yaw in radians."""
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("Invalid end-effector quaternion")
    x, y, z, w = q / norm
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(float(np.clip(sinp, -1.0, 1.0)))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
MAX_REQUEST_CHARS = 1024 * 1024


def _rpc_error(ident, code, message):
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": message}}


def rpc_response(request: Any, session: McpSession) -> Optional[Dict[str, Any]]:
    """Synchronous tools-only MCP subset. No cancellation or task capability."""
    if not isinstance(request, dict):
        return _rpc_error(None, -32600, "Request must be an object")
    ident = request.get("id")
    has_id = "id" in request
    if request.get("jsonrpc") != "2.0" or not isinstance(request.get("method"), str):
        return _rpc_error(None, -32600, "Invalid JSON-RPC request")
    if has_id and (type(ident) not in (str, int)):
        return _rpc_error(None, -32600, "Request ID must be a string or integer")
    # Notifications never dispatch tools, and never receive a response.
    if not has_id:
        return None
    params = request.get("params", {})
    if not isinstance(params, dict):
        return _rpc_error(ident, -32602, "Parameters must be an object")
    method = request["method"]
    if method == "initialize":
        version = params.get("protocolVersion")
        if not isinstance(version, str):
            return _rpc_error(ident, -32602, "protocolVersion must be a string")
        result = {
            "protocolVersion": version if version in SUPPORTED_PROTOCOL_VERSIONS else SUPPORTED_PROTOCOL_VERSIONS[0],
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "embodify-mcp", "version": __version__},
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": session.tools()}
    elif method == "resources/list":
        result = {"resources": []}
    elif method == "prompts/list":
        result = {"prompts": []}
    elif method == "tools/call":
        if not isinstance(params.get("name"), str) or not params["name"]:
            return _rpc_error(ident, -32602, "Tool name must be a nonempty string")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            return _rpc_error(ident, -32602, "Tool arguments must be an object")
        result = session.call_tool(params["name"], arguments)
    else:
        return _rpc_error(ident, -32601, "Method not found")
    return {"jsonrpc": "2.0", "id": ident, "result": result}


def serve_stdio(session: McpSession, stdin: Any = None, stdout: Any = None) -> None:
    source = stdin or sys.stdin
    sink = stdout or sys.stdout
    while True:
        line = source.readline(MAX_REQUEST_CHARS + 1)
        if not line:
            break
        if len(line) > MAX_REQUEST_CHARS:
            sink.write(json.dumps(_rpc_error(None, -32600, "Request too large")) + "\n")
            sink.flush()
            break  # Do not interpret an oversized request's tail as another action.
        if not line.strip():
            continue
        try:
            request = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))
        except (ValueError, RecursionError):
            response = _rpc_error(None, -32700, "Invalid JSON")
        else:
            try:
                response = rpc_response(request, session)
            except Exception as exc:
                print("[embodify-mcp] {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
                ident = request.get("id") if isinstance(request, dict) else None
                response = _rpc_error(ident, -32603, "Internal error; see operator logs")
        if response is not None:
            sink.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            sink.flush()


def _optional_float(value):
    return None if value is None else float(value)


BACKENDS = ("fake", "fake-two-arm", "libero", "robodojo", "remote")


def build_backend(args: argparse.Namespace) -> Backend:
    name = "fake" if args.fake else args.backend
    if name == "robodojo":
        from .backend.robodojo import RoboDojoBackend
        if not args.robodojo_config:
            raise ValueError("robodojo requires --robodojo-config")
        return RoboDojoBackend.from_config(args.robodojo_config)
    if name == "remote":
        from .backend.remote import RemoteBackend
        if not args.remote_config:
            raise ValueError("--backend remote requires --remote-config")
        return RemoteBackend.from_config(args.remote_config, frame_stride=args.frame_stride)
    if name == "fake":
        from .backend.fake import FakeBackend
        from .sim.fake import fake_tasks

        return FakeBackend(
            image_size=args.image_size,
            max_steps=args.max_steps,
            tasks=fake_tasks(4, suite=args.suite),
            task_source=lambda suite: fake_tasks(4, suite=suite),
            suite_source=lambda: [{"suite": args.suite, "n_tasks": 4}],
            default_task_index=args.task_index,
            init_state_index=args.init_state,
            controller_verified=args.controller_verified,
        )
    if name == "fake-two-arm":
        from .backend.fake import FakeTwoArmBackend

        return FakeTwoArmBackend(image_size=args.image_size, max_steps=args.max_steps)
    if name == "libero":
        from .backend.libero import LiberoBackend

        return LiberoBackend(
            image_size=args.image_size,
            max_steps=args.max_steps,
            suite=args.suite,
            task_index=args.task_index,
            init_state_index=args.init_state,
            renderer=args.renderer,
            controller_verified=args.controller_verified,
        )
    raise ValueError(f"Unknown backend {name!r}; options: {', '.join(BACKENDS)}")


def build_session(args: argparse.Namespace) -> McpSession:
    backend = build_backend(args)
    expected = getattr(args, "expected_episode_budget", None)
    if expected is not None and backend.info.episode_budget != expected:
        actual = backend.info.episode_budget
        backend.close()
        raise ValueError(f"Backend episode budget is {actual}, expected {expected}; sync the remote code and task configuration first")
    if hasattr(backend, "controller_params") and not args.controller_verified:
        print(
            "[embodify-mcp] The controller constants (max_position_delta_m / max_rotation_delta_rad /"
            " gripper_close_sign) have not been verified against this backend; this run is recorded as"
            " unverified. Add --controller-verified once they are.",
            file=sys.stderr,
        )
    return McpSession(
        backend,
        output_root=Path(args.output_root).expanduser(),
        max_steps_per_call=args.max_steps_per_call,
        task_locked=args.lock_task,
    )


def split_protocol_stdout() -> Any:
    """Move protocol output to its own fd, then point the process-level stdout at stderr.

    LIBERO's benchmark module prints `[info] using task orders ...` to stdout, and
    MuJoCo/OSMesa may write to fd 1 directly from C. Any non-JSON line breaks the host's
    JSON-RPC parsing, so the protocol stream needs an fd of its own.
    """
    # MCP uses UTF-8 regardless of the Windows console code page.
    for stream in (sys.stdin, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        protocol_fd = os.dup(sys.stdout.fileno())
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    except (AttributeError, OSError, io.UnsupportedOperation):
        return sys.stdout  # not a real file descriptor (such as a StringIO in tests); leave it alone
    sink = os.fdopen(protocol_fd, "w", buffering=1, encoding="utf-8", newline="\n")
    sys.stdout = sys.stderr
    return sink


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embodify MCP stdio server: robot observation and control tools")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--config",
        help="JSON settings file whose keys are the long options below; options given on the command line override it",
    )
    parser.add_argument("--backend", choices=BACKENDS, default="fake", help="Which backend to connect")
    parser.add_argument("--robodojo-config", help="RoboDojo worker configuration JSON")
    parser.add_argument("--remote-config", help="Backend transport JSON configuration")
    parser.add_argument("--expected-episode-budget", type=int, default=None,
                        help="Refuse startup when backend budget differs; does not override the backend")
    parser.add_argument("--frame-stride", type=int, default=None, help="Override remote image stride; all states are retained")
    parser.add_argument("--fake", action="store_true", help="Same as --backend fake: use FakeSim, do not load LIBERO")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-index", type=int, default=0, help="Default scene, used when the Agent does not choose one")
    parser.add_argument("--init-state", type=int, default=0, help="Default initial layout, used when the Agent does not choose one")
    parser.add_argument(
        "--lock-task",
        action="store_true",
        help=(
            "Fix the scene: refuse suite/task_index/init_state_index (or task) in reset_task. "
            "Required for controlled experiments, or the Agent could swap a hard task for an easier one."
        ),
    )
    parser.add_argument("--image-size", type=int, default=256)
    # LIBERO's own OffScreenRenderEnv defaults to a horizon of 1000 (bddl_base_domain.py), and the
    # adapter does not pass a horizon down, so 1000 is also the environment's hard limit. Raising it
    # requires changing the adapter first.
    parser.add_argument("--max-steps", type=int, default=1000, help="Simulation step budget per episode")
    parser.add_argument(
        "--max-steps-per-call",
        type=int,
        default=DEFAULT_MAX_STEPS_PER_CALL,
        help="Most internal simulation steps one tool call may run (a cap, not a quota)",
    )
    parser.add_argument("--renderer", default=None)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Where run logs and images are written")
    parser.add_argument(
        "--monitor-port",
        type=int,
        default=0,
        help="Also serve the read-only monitor page on this port (0 = off). It only reads --output-root and does not affect MCP",
    )
    parser.add_argument(
        "--controller-verified",
        action="store_true",
        help="Declare that the controller constants were verified by measurement; otherwise the journal records them as unverified",
    )
    return parser


#: Settings that name files; relative paths in a settings file are resolved against the file's folder.
_PATH_SETTINGS = {"robodojo_config", "remote_config", "output_root"}


def settings_argv(parser: argparse.ArgumentParser, path: str) -> List[str]:
    """Turn a JSON settings file into command-line arguments, so argparse checks them exactly like typed ones.

    The plugin starts the server with `--config ~/.embodify/config.json`; until an agent writes that file,
    the server runs on its defaults (the Fake backend).
    """
    file = Path(path).expanduser()
    if not file.is_file():
        print(f"[embodify-mcp] No settings file at {file}; using the defaults", file=sys.stderr)
        return []
    try:
        settings = json.loads(file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        parser.error(f"Cannot read settings file {file}: {exc}")
    if not isinstance(settings, dict):
        parser.error(f"Settings file {file} must hold a JSON object")
    options = {action.dest: action for action in parser._actions if action.option_strings}
    argv: List[str] = []
    for key, value in settings.items():
        dest = str(key).replace("-", "_")
        action = options.get(dest)
        if action is None or dest in ("config", "help", "version"):
            parser.error(f"Unknown setting {key!r} in {file}")
        flag = max(action.option_strings, key=len)
        if action.nargs == 0:
            if not isinstance(value, bool):
                parser.error(f"Setting {key!r} in {file} must be true or false")
            argv += [flag] if value else []
        elif value is not None:
            if dest in _PATH_SETTINGS:
                value = str(file.parent / Path(str(value)).expanduser())
            argv += [flag, str(value)]
    return argv


def parse_args(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line, with values from `--config` underneath it."""
    argv = list(sys.argv[1:] if argv is None else argv)
    config = parser.parse_known_args(argv)[0].config
    if not config:
        return parser.parse_args(argv)
    # Later occurrences win in argparse, so the command line overrides the settings file.
    return parser.parse_args(settings_argv(parser, config) + argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(build_parser(), argv)
    sink = split_protocol_stdout()
    session = build_session(args)
    if args.monitor_port:
        from .monitor import start_in_background

        start_in_background(Path(args.output_root).expanduser(), args.monitor_port)
    try:
        serve_stdio(session, stdout=sink)
    finally:
        session.close()


__all__ = [
    "McpSession",
    "McpToolError",
    "build_backend",
    "build_parser",
    "build_session",
    "rpc_response",
    "serve_stdio",
    "split_protocol_stdout",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    main()
