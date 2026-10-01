"""Two backends that run without any simulator installed, for tests and demos.

- `FakeBackend`: single arm, wraps `FakeSim` and uses exactly the same stepped loop as LIBERO
  (`stepped.py`). It exists so that all of the MCP layer can be tested without LIBERO.
- `FakeTwoArmBackend`: a purely kinematic backend with two arms and three cameras that does
  **not** go through `SteppedSimBackend` and implements `Backend` directly. It serves two
  purposes: testing the MCP layer's multi-arm / multi-camera paths (RoboDojo has two arms), and
  showing that the interface can be implemented without a stepped simulator (real robots and the
  RoboDojo bridge are of this kind).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from ..primitives.controller import (
    axis_angle_to_quat_xyzw,
    quat_xyzw_multiply,
    quat_xyzw_to_matrix,
    rotation_matrix_to_axis_angle,
)
from ..sim.base import SimAdapter, TaskSpec
from ..sim.fake import FakeSim, fake_tasks
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
from .catalogue import SuiteCatalogue, suite_texts
from .stepped import SteppedSimBackend, quat_angle_between

#: FakeSim cameras use LIBERO's public names, so the monitor page and tests treat both backends alike.
FAKE_CAMERAS = (
    ("agentview", "agentview", "Third person · agentview"),
    ("robot0_eye_in_hand", "eye_in_hand", "Wrist camera · eye in hand"),
)

#: Keys of FakeSim's ground-truth state (see `FakeSim.privileged_state`); none may leak to the Agent.
FAKE_FORBIDDEN = ("success", "privileged", "goal_position", "target_object", "distractor")


def generic_suite_texts(title: str) -> Any:
    """Suite-style catalogue texts that are not tied to any particular benchmark."""
    return suite_texts(
        reset_blurb="",
        suite_param="Switch to another suite; omit to keep the current one. See list_tasks (without arguments) for the suites.",
        list_description=(
            f"List the tasks {title} offers. **Independent of the current run**: callable at any time, renders no "
            "images, uses no step budget and does not change a running episode.\n"
            "Without arguments: returns suites, the name and task count of every suite, plus the total task count. "
            "Start here, pick a suite, then look inside it.\n"
            "With suite: returns task_index, instruction and n_init_states (the number of initial layouts of the "
            "same task) for every task in that suite.\n"
            "Pass the chosen suite and task_index to reset_task to start."
        ),
        list_suite_param="The suite whose tasks to list; omit to get only the suite overview.",
    )


class FakeBackend(SteppedSimBackend):
    """Single-arm fake backend. `adapter` may be a wrapped FakeSim (tests use that to record actions or script success)."""

    def __init__(
        self,
        adapter: Optional[SimAdapter] = None,
        *,
        image_size: int = 16,
        max_steps: int = 80,
        tasks: Optional[Sequence[TaskSpec]] = None,
        task_source: Optional[Callable[[str], Sequence[TaskSpec]]] = None,
        suite_source: Optional[Callable[[], Sequence[Dict[str, Any]]]] = None,
        default_task_index: int = 0,
        init_state_index: int = 0,
        controller_verified: bool = False,
    ) -> None:
        tasks = list(tasks) if tasks else fake_tasks(1)
        catalogue = SuiteCatalogue(
            texts=generic_suite_texts("FakeSim"),
            default_suite=tasks[0].suite,
            default_task_index=default_task_index,
            default_init_state=init_state_index,
            tasks=tasks,
            task_source=task_source,
            suite_source=suite_source,
        )
        super().__init__(
            adapter if adapter is not None else FakeSim(image_size=image_size, max_steps=max_steps),
            catalogue=catalogue,
            name="fake",
            title="FakeSim",
            server_name="fake-sim",
            frame="the FakeSim world frame (z up, table surface at z=0.80 m)",
            cameras=FAKE_CAMERAS,
            forbidden_terms=FAKE_FORBIDDEN,
            controller_verified=controller_verified,
        )


# -- two-arm kinematic backend -----------------------------------------------------

TWO_ARM_TABLE_Z = 0.80
#: How far and how much each step may move and turn at most. As in FakeSim, this only makes
#: "how many steps until it arrives" meaningful.
TWO_ARM_STEP_M = 0.05
TWO_ARM_STEP_RAD = 0.5
TWO_ARM_TOLERANCE_M = 0.008
TWO_ARM_TOLERANCE_RAD = 0.02
TWO_ARM_CAMERAS = (
    CameraInfo("cam_head", "Head camera · cam_head"),
    CameraInfo("cam_left_wrist", "Left wrist camera · cam_left_wrist"),
    CameraInfo("cam_right_wrist", "Right wrist camera · cam_right_wrist"),
)


class FakeTwoArmBackend(Backend):
    """Two arms (left / right) and three cameras. Each step moves the end effector a straight stretch toward the target, kept above the table.

    The right arm deliberately controls only rotation about z (incomplete, like a 5-DoF arm),
    to test how the session explains partially achievable orientation to the Agent.
    """

    def __init__(self, *, image_size: int = 16, max_steps: int = 80) -> None:
        self.image_size = int(image_size)
        self.max_steps = int(max_steps)
        tasks = [
            TaskSpec(suite="bimanual", task_index=i, name=f"bimanual_{i}",
                     instruction=f"hand the cup from the left gripper to the right one ({i})", n_init_states=2)
            for i in range(2)
        ]
        self._catalogue = SuiteCatalogue(
            texts=generic_suite_texts("FakeTwoArm"), default_suite="bimanual", tasks=tasks
        )
        self._arms: Dict[str, Dict[str, Any]] = {}
        self._step = 0
        self._task: Optional[TaskSpec] = None
        self._succeeded = False

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(
            name="fake_two_arm",
            title="FakeTwoArm",
            server_name="fake-two-arm",
            frame="the FakeTwoArm world frame (z up)",
            arms=(ArmInfo("left"), ArmInfo("right", rotation_axes=("z",))),
            cameras=TWO_ARM_CAMERAS,
            episode_budget=self.max_steps,
            forbidden_terms=("success", "privileged"),
            motion_text="Each step moves the end effector a straight stretch toward the target; returns on arrival or at the step limit.",
            gripper_text="The gripper opens or closes fully in one step.",
            move_stop_reasons=(("reached", "Within the target tolerance (8 mm in position, 0.02 rad in orientation)."),),
            gripper_stop_reasons=(("settled", "The gripper finished opening or closing."),),
        )

    @property
    def catalogue(self) -> Catalogue:
        return self._catalogue

    def reset(self, task: TaskSpec, variant: int) -> Snapshot:
        self._task = task
        self._step = 0
        self._succeeded = False
        offset = 0.02 * int(variant)
        self._arms = {
            "left": {"pos": np.array([0.0, 0.25 + offset, TWO_ARM_TABLE_Z + 0.2]), "quat": np.array([0.0, 0.0, 0.0, 1.0]), "close": False},
            "right": {"pos": np.array([0.0, -0.25, TWO_ARM_TABLE_Z + 0.2]), "quat": np.array([0.0, 0.0, 0.0, 1.0]), "close": False},
        }
        return self.observe()

    def observe(self) -> Snapshot:
        if self._task is None:
            raise RuntimeError("Reset before observing")
        n = self.image_size
        images = {}
        for k, cam in enumerate(TWO_ARM_CAMERAS):
            img = np.zeros((n, n, 3), dtype=np.uint8)
            img[..., k] = 60 + (self._step * 7) % 150
            images[cam.name] = img
        arms = {
            name: ArmState(
                pos=state["pos"].copy(),
                quat=state["quat"].copy(),
                gripper_opening=0.0 if state["close"] else 0.08,
                gripper_command="close" if state["close"] else "open",
            )
            for name, state in self._arms.items()
        }
        return Snapshot(step=self._step, arms=arms, images=images)

    def move_to(
        self,
        arm: str,
        pos: np.ndarray,
        quat: np.ndarray,
        *,
        max_steps: int,
        on_step: Optional[StepCallback] = None,
    ) -> MotionReport:
        state = self._arm(arm)
        target = np.asarray(pos, dtype=np.float64).reshape(3)
        target_quat = np.asarray(quat, dtype=np.float64).reshape(4)
        if arm == "right":  # controls only rotation about z; the other axes stay put
            target_quat = _keep_only_yaw(state["quat"], target_quat)
        executed = 0
        error = float(np.linalg.norm(target - state["pos"]))
        reason = LIMIT
        for _ in range(max_steps):
            delta = target - state["pos"]
            dist = float(np.linalg.norm(delta))
            if dist > TWO_ARM_STEP_M:
                delta = delta * (TWO_ARM_STEP_M / dist)
            state["pos"] = state["pos"] + delta
            state["pos"][2] = max(state["pos"][2], TWO_ARM_TABLE_Z + 0.005)
            state["quat"] = _rotate_toward(state["quat"], target_quat, TWO_ARM_STEP_RAD)
            self._step += 1
            executed += 1
            self._check_handover()  # update before the callback: the session polls success there
            snap = self.observe()
            if on_step is not None:
                on_step(snap)
            error = float(np.linalg.norm(target - state["pos"]))
            if error <= TWO_ARM_TOLERANCE_M and quat_angle_between(state["quat"], target_quat) <= TWO_ARM_TOLERANCE_RAD:
                reason = "reached"
                break
        return MotionReport(stop_reason=reason, executed_steps=executed, snapshot=self.observe(),
                            remaining_distance_m=error, telemetry={})

    def set_gripper(
        self,
        arm: str,
        close: bool,
        *,
        max_steps: int,
        on_step: Optional[StepCallback] = None,
    ) -> MotionReport:
        state = self._arm(arm)
        state["close"] = bool(close)
        executed = 0
        if max_steps > 0:
            self._step += 1
            executed = 1
            self._check_handover()  # update before the callback: the session polls success there
            if on_step is not None:
                on_step(self.observe())
        return MotionReport(stop_reason="settled" if executed else LIMIT, executed_steps=executed,
                            snapshot=self.observe(), telemetry={})

    def success(self) -> bool:
        return self._succeeded

    def end_episode(self) -> None:
        self._task = None

    def _arm(self, name: str) -> Dict[str, Any]:
        if name not in self._arms:
            raise ValueError(f"No arm named {name!r}")
        return self._arms[name]

    def _check_handover(self) -> None:
        """Both grippers closed and close enough together counts as a handover."""
        left, right = self._arms["left"], self._arms["right"]
        if left["close"] and right["close"] and float(np.linalg.norm(left["pos"] - right["pos"])) < 0.05:
            self._succeeded = True


def _rotate_toward(current: np.ndarray, target: np.ndarray, max_angle: float) -> np.ndarray:
    """Turn one step toward the target orientation, by at most max_angle radians."""
    rel = rotation_matrix_to_axis_angle(quat_xyzw_to_matrix(target) @ quat_xyzw_to_matrix(current).T)
    angle = float(np.linalg.norm(rel))
    if angle > max_angle:
        rel = rel * (max_angle / angle)
    return quat_xyzw_multiply(axis_angle_to_quat_xyzw(rel), current)


def _keep_only_yaw(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Keep only the rotation about z of the target relative to the current orientation."""
    rel = rotation_matrix_to_axis_angle(quat_xyzw_to_matrix(target) @ quat_xyzw_to_matrix(current).T)
    return quat_xyzw_multiply(axis_angle_to_quat_xyzw(np.array([0.0, 0.0, rel[2]])), current)
