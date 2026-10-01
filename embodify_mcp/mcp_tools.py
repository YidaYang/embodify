"""MCP tool table: names, parameters and descriptions.

Descriptions are assembled from the backend's self-description (`BackendInfo`) and the task
catalogue (`CatalogueTexts`): camera names and count, arms, frame wording, budget unit and stop
reasons are not hard-coded. The backend supplies the control criteria; the session adds budget
and repeated-call semantics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from .backend.base import COORDINATED_STOP_REASONS, BackendInfo, CatalogueTexts, join_names, number_word

#: Stop reasons the session reports itself (backend independent, all caused by budgets).
MOVE_SESSION_REASONS: Tuple[Tuple[str, str], ...] = (
    ("step_cap", "Hit the per-call internal step limit while the end effector was still moving toward the target. Call again with the remaining displacement in the same direction to continue."),
    ("budget", "The episode's total step budget ran out during the motion; only stop_episode can be called afterwards."),
)
GRIPPER_SESSION_REASONS: Tuple[Tuple[str, str], ...] = (
    ("step_cap", "Hit the per-call internal step limit before the opening was confirmed stable; the gripper may still be starting or moving. Repeat the same command to continue."),
    ("budget", "The episode's total step budget ran out during the action; only stop_episode can be called afterwards."),
    ("no_op", "The request matches the current gripper target and no gripper call is pending; nothing was executed and no budget was used. It does not mean an object was released."),
)

#: How each tool's result describes state. The wording differs slightly between tools, as it did before the refactor.
STATE_RESET = (
    "eef_pos_m is the end-effector XYZ position in meters; eef_rpy_rad.roll_x, pitch_y and yaw_z are the orientation "
    "about the robot base X/Y/Z axes in radians; gripper.command is the open/close target and gripper.opening_m the "
    "gripper opening in meters"
)
STATE_OBSERVE = "eef_pos_m is the end-effector XYZ position in meters, eef_rpy_rad the orientation about the base XYZ axes in radians, gripper holds the target command and opening_m in meters"
STATE_MOVE = "eef_pos_m is the end-effector XYZ position in meters, eef_rpy_rad the RPY orientation in radians, gripper holds command/opening_m"
STATE_GRIPPER = (
    "eef_pos_m is the end-effector XYZ position in meters, eef_rpy_rad the RPY orientation in radians, "
    "gripper.command the gripper target and gripper.opening_m the opening in meters"
)
STATE_INFO = "eef_pos_m is the XYZ position in meters, eef_rpy_rad the RPY orientation in radians, gripper holds command/opening_m"

_AXIS_NAMES = {"x": "X", "y": "Y", "z": "Z"}


def _reasons(items: Sequence[Tuple[str, str]]) -> str:
    return (
        f"action.stop_reason says why the call stopped, one of {number_word(len(items))}:\n"
        + "".join(f"- {name}: {text}\n" for name, text in items)
    )


class _Words:
    """Phrases for one backend, reused while assembling descriptions."""

    def __init__(self, info: BackendInfo) -> None:
        self.info = info
        self.unit = info.budget_unit
        self.multi = len(info.arms) > 1
        names = [camera.name for camera in info.cameras]
        plural = "" if len(names) == 1 else "s"
        pngs = f"{number_word(len(names))} separate PNG{plural}"
        self.images_full = f"images ({join_names(names)}: {pngs})"
        self.images_short = f"images ({pngs})"
        self.cameras = f"{number_word(len(names))} camera{plural}"

    def state(self, single: str) -> str:
        if not self.info.gripper_opening_measured:
            single += "; this backend cannot measure the gripper opening in meters, so opening_m is null, which does not mean a zero opening; command is only the control target"
        if not self.multi:
            return f"state ({single})"
        return (
            f"state (one entry per arm under arms, keyed {join_names(self.info.arm_names)}; for each arm, {single})"
        )

    def arm_property(self) -> Dict[str, Any]:
        return {
            "type": "string",
            "enum": list(self.info.arm_names),
            "description": f"The arm to act on: {join_names(self.info.arm_names, 'or')}.",
        }

    def rotation_note(self) -> str:
        """Some arms cannot control full orientation (5-DoF, for example); say so, or the Agent assumes every delta_rpy is honored."""
        limited = [arm for arm in self.info.arms if tuple(arm.rotation_axes) != ("x", "y", "z")]
        if not limited:
            return ""
        parts = []
        for arm in limited:
            axes = "/".join(_AXIS_NAMES.get(a, a) for a in arm.rotation_axes)
            noun = "axis" if len(arm.rotation_axes) == 1 else "axes"
            parts.append(f"{arm.name} can only rotate about the {axes} {noun}" if axes else f"{arm.name} controls position only, not orientation")
        return (
            "Not every arm has full orientation control: " + "; ".join(parts)
            + ". Components of delta_rpy an arm cannot control are ignored; trust the actual orientation in the returned state.\n"
        )


def build_tools(info: BackendInfo, texts: CatalogueTexts) -> List[Dict[str, Any]]:
    """All tools this backend exposes to the Agent."""
    w = _Words(info)
    unit = w.unit
    empty: Dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}

    move_props: Dict[str, Any] = {}
    grip_props: Dict[str, Any] = {}
    move_required = ["delta_xyz"]
    grip_required = ["state"]
    move_lead = "Move the end effector"
    if w.multi:
        move_props["arm"] = w.arm_property()
        grip_props["arm"] = w.arm_property()
        move_required = ["arm", "delta_xyz"]
        grip_required = ["arm", "state"]
        move_lead = "Move the end effector of the arm given by arm"
    move_props["delta_xyz"] = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
        "description": "[dx, dy, dz] in meters.",
    }
    move_props["delta_rpy"] = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
        "description": "[droll, dpitch, dyaw], increments about the robot base X/Y/Z axes in radians; omit to keep the current orientation.",
    }
    grip_props["state"] = {"type": "string", "enum": ["open", "close"]}

    tools = [
        {
            "name": "reset_task",
            "description": (
                f"Start a new {info.title} episode. "
                + texts.reset_intro
                + texts.reset_blurb
                + texts.reset_howto
                + "If an episode is already running, returns already_running; call stop_episode first.\n"
                + f"Returns run (short run ID), task ({texts.task_fields}), {w.images_full}, {w.state(STATE_RESET)}, "
                + f"step ({unit} used) and steps_left ({unit} left in this episode)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": texts.reset_properties,
                "required": [],
                "additionalProperties": False,
            },
        },
        {
            "name": "observe",
            "description": (
                f"Read the current RGB images from {w.cameras} and the end-effector/gripper state, without advancing "
                "the simulation or using the step budget. "
                f"Returns {w.images_full}, {w.state(STATE_OBSERVE)}, step ({unit} used), "
                f"steps_left ({unit} left) and status (running)."
            ),
            "inputSchema": empty,
        },
        {
            "name": "move_relative",
            "description": (
                f"{move_lead} by delta_xyz meters in {info.frame}, optionally rotating by delta_rpy about the robot base"
                " X/Y/Z axes (radians). "
                + info.motion_text + "\n"
                + _reasons(tuple(info.move_stop_reasons) + MOVE_SESSION_REASONS)
                + "Motion is a best-effort approach and may fall short of the requested delta: judge progress by "
                "action.achieved_delta_xyz (actual displacement of this call, meters) and action.remaining_distance_m "
                "(meters still missing to the requested target), and do not assume the request was achieved.\n"
                + w.rotation_note()
                + f"Returns {w.images_short}, {w.state(STATE_MOVE)}, action (the requested delta_xyz/delta_rpy, "
                "executed_steps, stop_reason, achieved_delta_xyz, remaining_distance_m), step"
                f" ({unit} used after the call), steps_left ({unit} left) and status (running). When the budget is "
                "already 0, returns a step_limit error instead of silently doing nothing."
            ),
            "inputSchema": {
                "type": "object",
                "properties": move_props,
                "required": move_required,
                "additionalProperties": False,
            },
        },
        {
            "name": "set_gripper",
            "description": (
                ("Open or close the gripper of the arm given by arm. " if w.multi else "Open or close the gripper. ")
                + info.gripper_text + "\n"
                + _reasons(tuple(info.gripper_stop_reasons) + GRIPPER_SESSION_REASONS)
                + f"Returns {w.images_short}, {w.state(STATE_GRIPPER)}, action (the requested"
                " gripper target, executed_steps, stop_reason, opening_m before and after), step"
                f" ({unit} used after the call), steps_left ({unit} left) and status (running). When the budget is"
                " already 0, returns a step_limit error and the gripper target stays unchanged."
            ),
            "inputSchema": {
                "type": "object",
                "properties": grip_props,
                "required": grip_required,
                "additionalProperties": False,
            },
        },
        {
            "name": "stop_episode",
            "description": "End the current simulation and save its frames, event log and summary. Returns run (short run ID), status (stopped) and artifacts (events: path of events.jsonl, frames: PNG frame directory, summary: path of summary.json).",
            "inputSchema": empty,
        },
        {
            "name": "get_session_info",
            "description": (
                "Read the state of **the current run**: task context, instruction, step count and robot state. Renders"
                " no images and uses no step budget; callable at any time, including before the first reset_task.\n"
                "This only says where you are now. To see which tasks are available, use list_tasks.\n"
                f"Returns run (short run ID), status (idle/running/stopped), step ({unit} used), "
                f"max_steps (the episode limit), steps_left ({unit} left), task ({texts.task_fields}), "
                "task_locked (true means the server fixed this run's scene, "
                "and reset_task accepts no scene arguments), and "
                f"{w.state(STATE_INFO)}. "
                f"While status is idle there is no task or state; instead it returns {texts.default_field} (the scene a"
                " reset_task without arguments starts)."
            ),
            "inputSchema": empty,
        },
        {
            "name": "list_tasks",
            "description": texts.list_description,
            "inputSchema": {
                "type": "object",
                "properties": texts.list_properties,
                "required": [],
                "additionalProperties": False,
            },
        },
    ]

    if info.supports_coordinated_control:
        arm_target = {
            "type": "object", "minProperties": 1, "additionalProperties": False,
            "properties": {
                "delta_xyz": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                              "description": "Relative to the end-effector position at the start of the call, in meters; omit to hold position."},
                "delta_rpy": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
                              "description": "RPY increment about the fixed scene X/Y/Z axes in radians; R_delta=Rz(yaw)Ry(pitch)Rx(roll), left-multiplied onto the starting orientation; omit to hold orientation."},
                "gripper": {"type": "string", "enum": ["open", "close"]},
            },
        }
        tools.append({
            "name": "control_arms",
            "description": (
                "Submit pose increments and optional gripper targets for one or more arms in a single call. Frame: " + info.frame + ". "
                "The remote controller advances all arms along a shared interpolation progress; a slower arm that lags holds "
                "its intermediate target, and the call completes only when every arm is within tolerance. "
                "Simultaneous arrival is not guaranteed. Arms not listed hold their pose from the start of the call; "
                "omitting a gripper target keeps the previous target. "
                "Gripper targets apply from the first environment action, without waiting for a grasp before moving; a "
                "gripper-only call executes one action, and command_applied does not mean stable or grasped. "
                "All arguments are validated before anything executes. "
                "Each joint environment action counts as one step, and the inner loop needs no per-step network round trip. "
                f"Returns {w.images_short}, the state of every arm, action.executed_steps and per-arm action.arms: "
                "achieved_delta_xyz, remaining_delta_xyz, remaining_distance_m, remaining_angle_rad, "
                "pose_reached and gripper_command_applied. Reaching a pose does not mean the object is grasped."
                "After step_cap, replan from the remaining displacement and the actual pose instead of repeating the full "
                "original delta; never replay an action automatically after a disconnect.\n"
                + _reasons(COORDINATED_STOP_REASONS + MOVE_SESSION_REASONS)
            ),
            "inputSchema": {
                "type": "object", "additionalProperties": False, "required": ["arms"],
                "properties": {"arms": {
                    "type": "object", "minProperties": 1, "additionalProperties": False,
                    "properties": {name: arm_target for name in info.arm_names},
                }},
            },
        })
    return tools
