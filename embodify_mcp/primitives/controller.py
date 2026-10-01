"""OSC delta controller: turns "bring the end effector to this pose" into a normalized 7-D action.

It is **scripted P control** with nothing learned and no demonstrations behind it; it could have
been written without ever seeing LIBERO.
"""

from __future__ import annotations

import numpy as np

from .params import ControllerParams


def position_action(
    current_pos: np.ndarray,
    target_pos: np.ndarray,
    *,
    gain: float,
    params: ControllerParams,
    action_dim: int = 7,
    close_gripper: bool | None = None,
) -> np.ndarray:
    """Take one step toward the target position.

    `close_gripper=None` keeps the current gripper command (expressed as 0, "no change").
    """
    cur = np.asarray(current_pos, dtype=np.float64).reshape(3)
    tgt = np.asarray(target_pos, dtype=np.float64).reshape(3)
    delta_m = (tgt - cur) * gain
    normalized = delta_m / params.max_position_delta_m
    action = np.zeros(action_dim, dtype=np.float64)
    action[:3] = np.clip(normalized, -1.0, 1.0)
    if close_gripper is not None:
        action[action_dim - 1] = params.gripper_close_sign * (1.0 if close_gripper else -1.0)
    return action


def rpy_xyz_to_matrix(rpy_rad: np.ndarray) -> np.ndarray:
    """Convert XYZ roll/pitch/yaw to Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = np.asarray(rpy_rad, dtype=np.float64).reshape(3)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion to a 3x3 rotation matrix."""
    x, y, z, w = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = np.linalg.norm([x, y, z, w])
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("quaternion must be finite and nonzero")
    x, y, z, w = np.asarray([x, y, z, w]) / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to an xyzw quaternion."""
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = 2.0 * np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-15))
            x, y, z, w = 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-15))
            x, y, z, w = (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s
        else:
            s = 2.0 * np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-15))
            x, y, z, w = (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s
    quat = np.asarray([x, y, z, w], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def quat_xyzw_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply xyzw quaternions, applying right then left."""
    x1, y1, z1, w1 = np.asarray(left, dtype=np.float64).reshape(4)
    x2, y2, z2, w2 = np.asarray(right, dtype=np.float64).reshape(4)
    out = np.asarray([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64)
    return out / np.linalg.norm(out)


def axis_angle_to_quat_xyzw(axis_angle: np.ndarray) -> np.ndarray:
    """Convert a rotation vector to an xyzw quaternion."""
    vec = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vec))
    if angle <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = vec / angle
    half = angle / 2.0
    return np.r_[axis * np.sin(half), np.cos(half)]


def rotation_matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    """Convert a relative rotation matrix to a rotation vector."""
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    angle = float(np.arccos(np.clip((np.trace(m) - 1.0) / 2.0, -1.0, 1.0)))
    if angle <= 1e-10:
        return np.zeros(3, dtype=np.float64)
    sine = np.sin(angle)
    if abs(sine) > 1e-7:
        axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) / (2.0 * sine)
        return axis * angle
    axis = np.sqrt(np.maximum((np.diag(m) + 1.0) / 2.0, 0.0))
    pivot = int(np.argmax(axis))
    if axis[pivot] <= 1e-8:
        return np.zeros(3, dtype=np.float64)
    for j in range(3):
        if j != pivot:
            axis[j] = (m[pivot, j] + m[j, pivot]) / (4.0 * axis[pivot])
    return axis * angle


def target_quat_from_rpy_delta(current_quat: np.ndarray, delta_rpy_rad: np.ndarray) -> np.ndarray:
    """Apply an XYZ RPY increment about robot-base axes to the current pose."""
    return matrix_to_quat_xyzw(rpy_xyz_to_matrix(delta_rpy_rad) @ quat_xyzw_to_matrix(current_quat))


def orientation_action(
    current_quat: np.ndarray,
    target_quat: np.ndarray,
    *,
    gain: float,
    params: ControllerParams,
    action_dim: int = 7,
) -> np.ndarray:
    """Return normalized OSC_POSE rotation-vector action toward a target quaternion."""
    current = quat_xyzw_to_matrix(current_quat)
    target = quat_xyzw_to_matrix(target_quat)
    rotvec = rotation_matrix_to_axis_angle(target @ current.T) * gain
    action = np.zeros(action_dim, dtype=np.float64)
    action[3:6] = np.clip(rotvec / params.max_rotation_delta_rad, -1.0, 1.0)
    return action


def gripper_action(close: bool, *, params: ControllerParams, action_dim: int = 7) -> np.ndarray:
    """Move only the gripper; the end effector stays put."""
    action = np.zeros(action_dim, dtype=np.float64)
    action[action_dim - 1] = params.gripper_close_sign * (1.0 if close else -1.0)
    return action


def saturation_fraction(action: np.ndarray) -> float:
    """The fraction of the three position axes that hit the action limit.

    It goes into the log: if most steps saturate, a near-zero success rate means the controller
    lacks authority, not that the model cannot do the task.
    """
    arr = np.asarray(action, dtype=np.float64).reshape(-1)[:3]
    return float(np.mean(np.abs(arr) >= 1.0 - 1e-9))
