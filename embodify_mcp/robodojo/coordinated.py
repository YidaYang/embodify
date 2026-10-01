"""Feedback-gated common-progress control; all physics executes on the worker thread."""
import math

import numpy as np

from ..backend.base import LIMIT, McpToolError, MotionReport
from ..backend.coordinated import validate_targets
from ..backend.stepped import quat_angle_between

POSITION_TOLERANCE = .008
ANGLE_TOLERANCE = .05
TRACK_POSITION_TOLERANCE = .004
TRACK_ANGLE_TOLERANCE = .025
MAX_TRANSLATION_STEP = .02
MAX_ROTATION_STEP = .1
STALL_WINDOW = 6
STALL_IMPROVEMENT = .0005


def slerp(start, end, progress):
    end = end if np.dot(start, end) >= 0 else -end
    dot = float(np.clip(np.dot(start, end), -1, 1))
    if dot > .9995:
        value = (1 - progress) * start + progress * end
    else:
        angle = math.acos(dot)
        value = (math.sin((1 - progress) * angle) * start + math.sin(progress * angle) * end) / math.sin(angle)
    return value / np.linalg.norm(value)


def control_arms(backend, targets, *, max_steps, on_step=None):
    names = backend.info.arm_names
    targets = validate_targets(targets, names)
    if type(max_steps) is not int or max_steps < 0:
        raise McpToolError("invalid_input", "Invalid action budget")
    snapshot = backend.observe()
    start_step = snapshot.step
    starts = snapshot.arms
    goals = {name: targets.get(name, {"pos": starts[name].pos.copy(), "quat": starts[name].quat.copy()})
             for name in names}
    commands = dict(backend._commands)
    for name, goal in goals.items():
        if "gripper" in goal:
            commands[name] = goal["gripper"]
    changed_gripper = commands != backend._commands
    deltas = {n: goals[n]["pos"] - starts[n].pos for n in names}
    angles = {n: quat_angle_between(starts[n].quat, goals[n]["quat"]) for n in names}
    distances = {n: float(np.linalg.norm(deltas[n])) for n in names}
    if not all(math.isfinite(v) for v in (*distances.values(), *angles.values())):
        raise McpToolError("invalid_input", "Joint target out of numeric range")
    needs_motion = any(distances[n] > POSITION_TOLERANCE or angles[n] > ANGLE_TOLERANCE for n in names)
    allowed = min(max_steps, max(0, backend.info.episode_budget - start_step))
    progress, reason = 0., LIMIT
    trace = []
    histories = {n: [] for n in names}

    def waypoints(s):
        return {n: (starts[n].pos + s * deltas[n], slerp(starts[n].quat, goals[n]["quat"], s)) for n in names}

    def residuals(points):
        return {n: (float(np.linalg.norm(points[n][0] - snapshot.arms[n].pos)),
                    quat_angle_between(points[n][1], snapshot.arms[n].quat)) for n in names}

    def reached(errors, pos_tol, rot_tol):
        return all(d <= pos_tol and a <= rot_tol for d, a in errors.values())

    if backend.driver.ended:
        reason = "ended"
    elif allowed == 0:
        reason = LIMIT
    elif not needs_motion and not changed_gripper:
        reason = "no_op"
    else:
        # Reserve the tracking tolerance so a new waypoint remains within the
        # per-action displacement bound even when the measured pose lags behind.
        increment = min([1.] + [(MAX_TRANSLATION_STEP - TRACK_POSITION_TOLERANCE) / d
                                for d in distances.values() if d > 0]
                        + [(MAX_ROTATION_STEP - TRACK_ANGLE_TOLERANCE) / a
                           for a in angles.values() if a > 0])
        for _ in range(allowed):
            if backend.driver.ended:
                reason = "ended"
                break
            current_errors = residuals(waypoints(progress))
            if progress < 1 and reached(current_errors, TRACK_POSITION_TOLERANCE, TRACK_ANGLE_TOLERANCE):
                progress = min(1., progress + increment) if needs_motion else 1.
                histories = {n: [] for n in names}
            points = waypoints(progress)
            action = {}
            for name in names:
                pos, quat = points[name]
                # A contact disturbance may push an arm outside the tracking
                # tolerance. Recover toward the fixed waypoint with bounded IK
                # targets, without advancing the other arm's shared progress.
                measured = snapshot.arms[name]
                offset = pos - measured.pos
                distance = float(np.linalg.norm(offset))
                pos = measured.pos + offset * min(1., MAX_TRANSLATION_STEP / max(distance, 1e-12))
                angle = quat_angle_between(measured.quat, quat)
                quat = slerp(measured.quat, quat, min(1., MAX_ROTATION_STEP / max(angle, 1e-12)))
                action[name + "_ee_pose"] = list(pos) + list(quat[[3, 0, 1, 2]])
                action[name + "_ee_joint_state"] = [0. if commands[name] == "close" else 1.]
            backend._commands = commands.copy()
            snapshot = backend._execute(action, on_step)
            trace.append(progress)
            errors = residuals(points)
            if not needs_motion:
                reason = "command_applied"
                break
            if progress == 1 and reached(errors, POSITION_TOLERANCE, ANGLE_TOLERANCE):
                reason = "reached"
                break
            if backend.driver.ended:
                reason = "ended"
                break
            pos_tol = POSITION_TOLERANCE if progress == 1 else TRACK_POSITION_TOLERANCE
            rot_tol = ANGLE_TOLERANCE if progress == 1 else TRACK_ANGLE_TOLERANCE
            stalled = False
            for name, (distance, angle) in errors.items():
                history = histories[name]
                history.append(distance + .1 * angle)
                del history[:-STALL_WINDOW]
                if (distance > pos_tol or angle > rot_tol) and len(history) == STALL_WINDOW:
                    stalled |= history[0] - history[-1] < STALL_IMPROVEMENT
            if stalled:
                reason = "stalled"
                break
    final_errors = residuals(waypoints(1.))
    executed = snapshot.step - start_step
    feedback = {n: {"remaining_distance_m": d, "remaining_angle_rad": a,
                    "pose_reached": d <= POSITION_TOLERANCE and a <= ANGLE_TOLERANCE,
                    "gripper_command_applied": "gripper" in goals[n] and executed > 0}
                for n, (d, a) in final_errors.items()}
    return MotionReport(reason, executed, snapshot, max(d for d, _ in final_errors.values()),
                        {"common_progress": trace, "position_tolerance_m": POSITION_TOLERANCE,
                         "angle_tolerance_rad": ANGLE_TOLERANCE}, feedback)
