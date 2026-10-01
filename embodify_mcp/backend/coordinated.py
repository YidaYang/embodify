"""Validation shared by the coordinated backend API and its transport."""
import math

import numpy as np

from .base import McpToolError
from .wire import vector


def validate_targets(targets, arm_names):
    """Normalize all absolute goals before a backend changes any command latch."""
    try:
        if not isinstance(targets, dict) or not targets or set(targets) - set(arm_names):
            raise ValueError("unknown or empty arms")
        result = {}
        for name, value in targets.items():
            if not isinstance(value, dict) or set(value) - {"pos", "quat", "gripper"}:
                raise ValueError("unknown target fields")
            pos, quat = vector(value["pos"], 3).copy(), vector(value["quat"], 4).copy()
            norm = float(np.linalg.norm(quat))
            if not math.isfinite(norm) or norm < 1e-9:
                raise ValueError("invalid quaternion")
            item = {"pos": pos, "quat": quat / norm}
            if "gripper" in value:
                if value["gripper"] not in ("open", "close"):
                    raise ValueError("invalid gripper")
                item["gripper"] = value["gripper"]
            result[name] = item
        return result
    except (ValueError, TypeError, KeyError, OverflowError):
        raise McpToolError("invalid_input", "Joint targets need valid arm names, finite poses and optional open/close gripper targets") from None


def validate_arm_results(value, arm_names):
    fields = {"remaining_distance_m", "remaining_angle_rad", "pose_reached", "gripper_command_applied"}
    if not isinstance(value, dict) or set(value) != set(arm_names):
        raise ValueError("invalid coordinated arm results")
    for item in value.values():
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("invalid coordinated feedback fields")
        for key in ("remaining_distance_m", "remaining_angle_rad"):
            if (type(item[key]) not in (int, float) or not math.isfinite(item[key]) or item[key] < 0):
                raise ValueError("invalid coordinated residual")
        for key in ("pose_reached", "gripper_command_applied"):
            if type(item[key]) is not bool:
                raise ValueError("invalid coordinated flag")
    return value
