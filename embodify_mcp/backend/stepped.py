"""Backend base for stepped simulators (`SimAdapter`: one normalized 7-D OSC delta action per step).

The P-control loop that brings the end effector to a target pose used to live in the MCP layer
(`move_relative` in `mcp.py`) and now lives here: LIBERO and FakeSim are both stepped
simulators and share this one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..obs import Observation
from ..primitives.controller import (
    gripper_action,
    orientation_action,
    position_action,
    quat_xyzw_to_matrix,
    rotation_matrix_to_axis_angle,
)
from ..primitives.params import ControllerParams
from ..sim.base import SimAdapter, TaskSpec
from .base import (
    LIMIT,
    ArmInfo,
    ArmState,
    Backend,
    BackendInfo,
    CameraInfo,
    Catalogue,
    MotionReport,
    Snapshot,
    StepCallback,
)

#: The most internal simulation steps one tool call may run (the MCP session default, see `mcp.py`).
#:
#: This is a **cap**, not a quota: the loop stops early once it arrives or stalls. The old
#: implementation always ran exactly 8 steps, so it fell short (a measured 5 cm request covered
#: only 74% in 8 steps), burned budget after already arriving, and, worst of all, "blocked" and
#: "controller too weak" looked identical in its results.
#:
#: 30 is based on measurements on real LIBERO (2026-09-20): a 5 cm move converged to the 8 mm
#: tolerance in 11 steps, and a 2 cm move plus 0.2 rad rotation in 14 steps. 30 leaves ample
#: margin while keeping one call's wall-clock time around 6.6 s (measured 0.22 s per step with
#: two 256x256 cameras).
DEFAULT_MAX_STEPS_PER_CALL = 30

#: Arrival criteria. These belong to the MCP loop and **deliberately do not reuse
#: `PrimitiveParams`**: primitive parameters are tuned on dev tasks and then frozen, and the MCP
#: loop should not drift with them (both position tolerances happen to be 8 mm today, but
#: changing either side does not affect the other).
POSITION_TOLERANCE_M = 0.008
ROTATION_TOLERANCE_RAD = 0.02

#: Stall criterion: the target is not reached, yet the end effector barely moved for
#: STALL_WINDOW consecutive steps.
#:
#: While the position error exceeds the tolerance, the P controller always sends a nonzero
#: command, so "a sizeable command but no motion" can only mean something is blocking it. The
#: threshold applies to the **total** displacement over the window.
#:
#: Displacement and rotation must **both** fall below their thresholds: a wrist that is still
#: turning means the controller is still acting, so it is not stuck. On real LIBERO
#: (2026-09-20, pressing the end effector 0.20 m into the table), per-step displacement decayed
#: from 13 mm to 0.64 / 0.38 / 0.23 / 0.16 mm and the residual froze at 9.4 mm. The displacement
#: criterion held from step 24, but contact-induced wrist rotation delayed the trigger to step 30.
#: Both thresholds are therefore measured, though STALL_EPSILON_RAD is on the tight side; the
#: journal's step_rotations_rad is the data for tuning it next.
STALL_WINDOW = 3
STALL_EPSILON_M = 0.001
STALL_EPSILON_RAD = 0.005

#: The Panda's actuator target changes by only 0.01 per physics substep; at 20 Hz / 0.002 s a
#: full reversal takes 8 control steps. Two more steps of observation keep a not-yet-started
#: gripper from being mistaken for a settled one. This is a startup guard for the current stepped
#: simulator backends, not a guarantee for any hardware; the budget remains a hard cap, and
#: repeated calls accumulate steps for the same command.
GRIPPER_MIN_COMMAND_STEPS = 10
#: After startup, check the opening range over the whole window rather than per-step changes
#: below a threshold (the latter would take a steady slow movement for settled).
GRIPPER_SETTLE_WINDOW = 2
GRIPPER_SETTLE_EPSILON_M = 0.0005

#: P-control gain. Must be < 1, or the displacement at small errors exceeds the error itself and oscillates around the target.
GAIN = 0.6

MOTION_TEXT = "The server approaches the target step by step with P control and returns only when a definite stop condition triggers, not after a fixed number of steps."
GRIPPER_TEXT = "The server keeps sending the command and watches the opening width, and returns only when a definite stop condition triggers, not after a fixed number of steps."
MOVE_STOP_REASONS: Tuple[Tuple[str, str], ...] = (
    ("reached", "Within the target tolerance (8 mm in position, 0.02 rad in orientation); the motion is complete."),
    (
        "stalled",
        "Barely moved for 3 consecutive steps (less than 1 mm in total) without reaching the target, which usually "
        "means something is blocking it: the table, an object, or a joint limit. Repeating the same command has no "
        "effect; change direction, or lift or back off first. This is the main signal of contact.",
    ),
)
GRIPPER_STOP_REASONS: Tuple[Tuple[str, str], ...] = (
    (
        "settled",
        "After at least 10 internal steps of the same command, the opening varied by less than 0.5 mm over the last "
        "2 steps. This only means the opening is momentarily stable, possibly because something blocks it; it does "
        "not guarantee a fully open gripper, a firm grasp or a released object. Check state.gripper.opening_m and the images.",
    ),
)

#: Stepped simulators are all single-arm; this is the name of that one arm.
ARM = "robot0"


def quat_angle_between(first: np.ndarray, second: np.ndarray) -> float:
    """The angle between two xyzw orientations, in radians.

    Reuses the rotation math in `controller.py` instead of a second copy: the pose error and
    the stall criterion both depend on it, and duplicate implementations drift apart quietly.
    """
    rel = quat_xyzw_to_matrix(second) @ quat_xyzw_to_matrix(first).T
    return float(np.linalg.norm(rotation_matrix_to_axis_angle(rel)))


class SteppedSimBackend(Backend):
    """Wraps a `SimAdapter` into an action-level backend.

    `cameras` holds (public name, simulator name, human-readable label) triples: the public name
    is what the Agent and archived frames use, the simulator name is the key in the simulator's
    observation. If the simulator name is missing, the public name is tried.
    """

    def __init__(
        self,
        adapter: SimAdapter,
        *,
        catalogue: Catalogue,
        name: str,
        title: str,
        server_name: str,
        frame: str,
        cameras: Sequence[Tuple[str, str, str]],
        forbidden_terms: Sequence[str] = (),
        controller_verified: bool = False,
    ) -> None:
        self.adapter = adapter
        self._catalogue = catalogue
        self._name = name
        self._title = title
        self._server_name = server_name
        self._frame = frame
        self._cameras = tuple((str(a), str(b), str(c)) for a, b, c in cameras)
        self._forbidden = tuple(forbidden_terms)
        # One source of truth for the controller constants. `verified_against_facts` must not be
        # asserted here: C1 left gripper_close_sign unmeasured, so the honest default is False and
        # the caller has to opt in after actually probing the simulator.
        self.controller_params = ControllerParams(verified_against_facts=bool(controller_verified))
        #: The gripper target command. It rides along on every motion step, so it is controller
        #: state and lives in the backend.
        self.gripper_command = "open"
        self._gripper_command_steps = 0
        self._gripper_width_window: List[float] = []
        self._last_obs: Optional[Observation] = None

    # -- self-description --------------------------------------------------

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            name=self._name,
            title=self._title,
            server_name=self._server_name,
            frame=self._frame,
            arms=(ArmInfo(ARM),),
            cameras=tuple(CameraInfo(public, label) for public, _, label in self._cameras),
            episode_budget=int(self.adapter.max_steps),
            forbidden_terms=self._forbidden,
            motion_text=MOTION_TEXT,
            gripper_text=GRIPPER_TEXT,
            move_stop_reasons=MOVE_STOP_REASONS,
            gripper_stop_reasons=GRIPPER_STOP_REASONS,
        )

    @property
    def catalogue(self) -> Catalogue:
        return self._catalogue

    def journal_params(self) -> Dict[str, Any]:
        params = self.controller_params
        return {
            "max_position_delta_m": params.max_position_delta_m,
            "max_rotation_delta_rad": params.max_rotation_delta_rad,
            "gripper_close_sign": params.gripper_close_sign,
            "verified_against_facts": params.verified_against_facts,
            "position_tolerance_m": POSITION_TOLERANCE_M,
            "rotation_tolerance_rad": ROTATION_TOLERANCE_RAD,
            "stall_window": STALL_WINDOW,
            "stall_epsilon_m": STALL_EPSILON_M,
            "gripper_min_command_steps": GRIPPER_MIN_COMMAND_STEPS,
            "gripper_settle_window": GRIPPER_SETTLE_WINDOW,
            "gripper_settle_epsilon_m": GRIPPER_SETTLE_EPSILON_M,
        }

    # -- lifecycle ---------------------------------------------------------

    def reset(self, task: TaskSpec, variant: int) -> Snapshot:
        self.gripper_command = "open"
        self._gripper_command_steps = 0
        obs = self.adapter.reset(task, variant)
        self._gripper_width_window = [float(obs.gripper_width)]
        return self._snapshot(obs)

    def observe(self) -> Snapshot:
        return self._snapshot(self.adapter.observe())

    def success(self) -> bool:
        return bool(self.adapter.success())

    def end_episode(self) -> None:
        self.adapter.close()
        self._last_obs = None

    # -- actions -----------------------------------------------------------

    def move_to(
        self,
        arm: str,
        pos: np.ndarray,
        quat: np.ndarray,
        *,
        max_steps: int,
        on_step: Optional[StepCallback] = None,
    ) -> MotionReport:
        self._check_arm(arm)
        obs = self._current()
        target = np.asarray(pos, dtype=np.float64).reshape(3)
        target_quat = np.asarray(quat, dtype=np.float64).reshape(4)
        params = self.controller_params
        action_dim = self.adapter.action_dim
        saturations: List[float] = []
        # The latched gripper command rides along on every motion step. Its sign comes from the
        # controller params, exactly like set_gripper: a hardcoded +1 here would silently keep the
        # old convention if C1 ever finds the sign is inverted, and the hand would open mid-motion.
        gripper_value = float(
            gripper_action(self.gripper_command == "close", params=params, action_dim=action_dim)[-1]
        )

        executed = 0
        # Per-step displacement and residual error. These are the calibration record: the step cap
        # and the stall threshold were picked from one data point, and this journal field is what
        # lets the first real episode replace the guess with a measurement.
        step_distances: List[float] = []
        step_rotations: List[float] = []
        step_errors: List[float] = []
        recent_pos: List[float] = []
        recent_rot: List[float] = []
        pos_error = float(np.linalg.norm(target - np.asarray(obs.eef_pos, dtype=np.float64).reshape(3)))
        stop_reason = LIMIT
        for _ in range(max_steps):
            action = position_action(
                obs.eef_pos,
                target,
                gain=GAIN,
                params=params,
                action_dim=action_dim,
                close_gripper=None,
            )
            action[3:6] = orientation_action(
                obs.eef_quat,
                target_quat,
                gain=GAIN,
                params=params,
                action_dim=action_dim,
            )[3:6]
            action[-1] = gripper_value
            saturations.append(float(np.mean(np.abs(action[:3]) >= 1.0 - 1e-9)))
            before_pos = np.asarray(obs.eef_pos, dtype=np.float64).reshape(3).copy()
            before_quat = np.asarray(obs.eef_quat, dtype=np.float64).reshape(4).copy()
            # `step` already returns the post-step observation, so the loop must not call `observe`
            # again: that doubled the rendering cost of every internal step for nothing.
            obs = self._step(action)
            executed += 1
            if on_step is not None:
                on_step(self._snapshot(obs, remember=False))

            now_pos = np.asarray(obs.eef_pos, dtype=np.float64).reshape(3)
            moved = float(np.linalg.norm(now_pos - before_pos))
            turned = quat_angle_between(before_quat, obs.eef_quat)
            pos_error = float(np.linalg.norm(target - now_pos))
            rot_error = quat_angle_between(obs.eef_quat, target_quat)
            step_distances.append(moved)
            step_rotations.append(turned)
            step_errors.append(pos_error)
            recent_pos.append(moved)
            recent_rot.append(turned)
            del recent_pos[:-STALL_WINDOW]
            del recent_rot[:-STALL_WINDOW]

            if pos_error <= POSITION_TOLERANCE_M and rot_error <= ROTATION_TOLERANCE_RAD:
                stop_reason = "reached"
                break
            if (
                len(recent_pos) == STALL_WINDOW
                and sum(recent_pos) < STALL_EPSILON_M
                and sum(recent_rot) < STALL_EPSILON_RAD
            ):
                stop_reason = "stalled"
                break

        return MotionReport(
            stop_reason=stop_reason,
            executed_steps=executed,
            snapshot=self._snapshot(obs, remember=False),
            remaining_distance_m=pos_error,
            telemetry={
                "step_distances_m": step_distances,
                "step_rotations_rad": step_rotations,
                "step_errors_m": step_errors,
                "mean_saturation": float(np.mean(saturations)) if saturations else 0.0,
            },
        )

    def set_gripper(
        self,
        arm: str,
        close: bool,
        *,
        max_steps: int,
        on_step: Optional[StepCallback] = None,
    ) -> MotionReport:
        self._check_arm(arm)
        obs = self._current()
        widths = [float(obs.gripper_width)]
        command = "close" if close else "open"
        if command != self.gripper_command:
            self._gripper_command_steps = 0
            self._gripper_width_window = [float(obs.gripper_width)]
        self.gripper_command = command
        params = self.controller_params
        stop_reason = LIMIT
        steps = 0
        for _ in range(max_steps):
            action = gripper_action(close, params=params, action_dim=self.adapter.action_dim)
            obs = self._step(action)
            steps += 1
            if on_step is not None:
                on_step(self._snapshot(obs, remember=False))
            widths.append(float(obs.gripper_width))
            # A quiet startup is not completion. Keep history across capped calls, so even
            # one-step calls can confirm stability without restarting the command ramp.
            window = self._gripper_width_window
            if (
                self._gripper_command_steps >= GRIPPER_MIN_COMMAND_STEPS
                and len(window) > GRIPPER_SETTLE_WINDOW
                and max(window) - min(window) < GRIPPER_SETTLE_EPSILON_M
            ):
                stop_reason = "settled"
                break
        return MotionReport(
            stop_reason=stop_reason,
            executed_steps=steps,
            snapshot=self._snapshot(obs, remember=False),
            telemetry={"openings_m": widths},
        )

    # -- internals ---------------------------------------------------------

    def _check_arm(self, arm: str) -> None:
        if arm != ARM:
            raise ValueError(f"No arm named {arm!r}; this backend only has {ARM!r}")

    def _current(self) -> Observation:
        """The observation at the start of an action.

        The session has just observed before every action, and a stepped simulator's state only
        changes through step, so that result is reused instead of rendering again.
        """
        if self._last_obs is None:
            self._last_obs = self.adapter.observe()
        return self._last_obs

    def _step(self, action: np.ndarray) -> Observation:
        obs = self.adapter.step(action)
        self._gripper_command_steps += 1
        self._gripper_width_window.append(float(obs.gripper_width))
        self._gripper_width_window = self._gripper_width_window[-(GRIPPER_SETTLE_WINDOW + 1) :]
        self._last_obs = obs
        return obs

    def _snapshot(self, obs: Observation, *, remember: bool = True) -> Snapshot:
        if remember:
            self._last_obs = obs
        images: Dict[str, np.ndarray] = {}
        for public, internal, _ in self._cameras:
            image = obs.rgb.get(internal)
            if image is None and public != internal:
                image = obs.rgb.get(public)
            if image is not None:
                images[public] = image
        return Snapshot(
            step=int(obs.step_index),
            arms={
                ARM: ArmState(
                    pos=np.asarray(obs.eef_pos, dtype=np.float64).reshape(3),
                    quat=np.asarray(obs.eef_quat, dtype=np.float64).reshape(4),
                    gripper_opening=float(obs.gripper_width),
                    gripper_command=self.gripper_command,
                )
            },
            images=images,
        )
