"""Action-level RoboDojo adapter. Driver operations run on the simulator thread."""
from __future__ import annotations
import numpy as np
from ..backend.base import (Backend, BackendInfo, ArmInfo, CameraInfo, ArmState,
                            Snapshot, MotionReport, LIMIT, BackendDisconnected, McpToolError)
from ..backend.stepped import quat_angle_between
from ..backend.wire import vector
from .catalogue import TaskCatalogue
from .config import ARMS, CAMERAS, validate_episode_budget


def info(episode_budget):
    validate_episode_budget(episode_budget)
    return BackendInfo(
        name='robodojo', title='RoboDojo · ARX X5', server_name='robodojo-sim',
        frame=("the RoboDojo scene frame: origin at the current environment origin, XYZ parallel to the world axes, "
               "z up; not the base frame of either arm"),
        arms=tuple(ArmInfo(a) for a in ARMS),
        cameras=tuple(CameraInfo(c, c) for c in CAMERAS),
        episode_budget=episode_budget, budget_unit='environment action calls', has_success=True,
        forbidden_terms=('success', 'privileged', 'reward', 'score', 'scene_layout', 'seed', 'object_pose'),
        motion_text=('The control loop runs on the simulator side. Each internal step calls one end-effector action of the '
                     'environment; both arms receive full targets, and the arm not selected holds its pose.'),
        gripper_text=('Sends the open/close target and executes one environment action; command_applied only means the command '
                      'was executed, not that the gripper is stable. The opening in meters is unknown.'),
        move_stop_reasons=(('reached', 'The pose error is within the threshold.'),
                           ('stalled', 'No progress toward the target over several environment actions.'),
                           ('ended', 'The environment stopped accepting actions.')),
        gripper_stop_reasons=(('command_applied', 'The environment executed the open/close command; this does not mean the mechanism is stable.'),
                              ('ended', 'The environment stopped accepting actions.')),
        gripper_opening_measured=False, supports_coordinated_control=True)


def decode_observation(raw, step, commands):
    """Whitelist only poses and RGB; never copy environment metadata to Snapshot."""
    arms, images = {}, {}
    for arm in ARMS:
        pose = vector(raw['state'][arm + '_ee_pose'], 7)
        q = pose[3:]
        norm = np.linalg.norm(q)
        if norm < 1e-9:
            raise ValueError('invalid quaternion')
        q = q / norm  # upstream wxyz -> our xyzw
        arms[arm] = ArmState(pose[:3].copy(), q[[1,2,3,0]], None, commands[arm])
    for camera in CAMERAS:
        rgb = np.asarray(raw['vision'][camera]['color'])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] not in (3, 4) or min(rgb.shape[:2]) < 1:
            raise ValueError('invalid RGB camera')
        images[camera] = np.ascontiguousarray(rgb[:,:,:3]).copy()
    return Snapshot(step, arms, images)


def target_action(snapshot, commands, arm=None, pos=None, quat=None):
    result = {}
    for name in ARMS:
        state = snapshot.arms[name]
        p = state.pos if name != arm or pos is None else vector(pos, 3)
        q = state.quat if name != arm or quat is None else vector(quat, 4)
        q = q / np.linalg.norm(q)
        result[name + '_ee_pose'] = list(p) + list(q[[3,0,1,2]])
        result[name + '_ee_joint_state'] = [0.0 if commands[name] == 'close' else 1.0]
    return result


class EpisodeBackend(Backend):
    def __init__(self, driver):
        self.driver = driver
        # One worker runs one task: the one its driver was started for.
        self._catalogue = TaskCatalogue(driver.task, episode_budget_override=driver.budget, offered=(driver.task,))
        self._info = info(driver.budget)
        self._snapshot = None
        self._step = 0
        self._commands = {a: 'open' for a in ARMS}
        self._audit = None

    @property
    def info(self): return self._info
    @property
    def catalogue(self): return self._catalogue

    def _accept(self, raw):
        self._snapshot = decode_observation(raw, self._step, self._commands)
        return self._snapshot

    def reset(self, task, variant):
        self.catalogue.commit(task, variant)
        if self._snapshot is not None:
            raise McpToolError('reset_requires_worker', 'Start a new simulator process before reset')
        raw = self.driver.reset()
        if self.driver.budget != self.info.episode_budget:
            raise RuntimeError('Unexpected task action budget')
        self.catalogue.accept_instruction(raw.get('instruction'))
        for arm in ARMS:
            value = vector(raw['state'][arm + '_ee_joint_state'], 1)[0]
            if not 0 <= value <= 1: raise ValueError('invalid gripper target')
            self._commands[arm] = 'close' if value < .5 else 'open'
        return self._accept(raw)

    def observe(self):
        if self._snapshot is None: raise McpToolError('not_running', 'Call reset_task first')
        # No world step, and fresh camera capture follows the official renderer sync.
        return self._accept(self.driver.observe())

    def _execute(self, action, on_step):
        if self.driver.ended: raise McpToolError('episode_ended', 'The environment has stopped; call stop_episode first')
        raw = self.driver.step(action)
        self._step += 1
        snapshot = self._accept(raw)
        if on_step: on_step(snapshot)
        return snapshot

    def move_to(self, arm, pos, quat, *, max_steps, on_step=None):
        if arm not in ARMS: raise McpToolError('invalid_input', 'Unknown arm')
        pos, quat = vector(pos,3), vector(quat,4)
        if np.linalg.norm(quat) < 1e-9: raise McpToolError('invalid_input', 'Invalid orientation')
        quat = quat / np.linalg.norm(quat)
        start, reason, errors = self._step, LIMIT, []
        snapshot = self.observe()
        for _ in range(max(0, min(max_steps, self.info.episode_budget-self._step))):
            if self.driver.ended:
                reason = 'ended'; break
            state = snapshot.arms[arm]
            delta = pos - state.pos
            distance = float(np.linalg.norm(delta))
            angle = quat_angle_between(quat, state.quat)
            if distance <= .008 and angle <= .05:
                reason = 'reached'; break
            # Limit each IK target; the upstream client does joint interpolation.
            p = state.pos + delta * min(1., .02 / max(distance, 1e-12))
            q = quat if np.dot(quat,state.quat) >= 0 else -quat
            fraction = min(1., .1 / max(angle,1e-12))
            q = state.quat * (1-fraction) + q * fraction
            q /= np.linalg.norm(q)
            snapshot = self._execute(target_action(snapshot,self._commands,arm,p,q),on_step)
            errors.append(float(np.linalg.norm(pos-snapshot.arms[arm].pos)) + .1*quat_angle_between(quat,snapshot.arms[arm].quat))
            if len(errors) >= 6 and errors[-6] - errors[-1] < .0005:
                reason = 'stalled'; break
        distance = float(np.linalg.norm(pos-snapshot.arms[arm].pos))
        if distance <= .008 and quat_angle_between(quat,snapshot.arms[arm].quat) <= .05:
            reason = 'reached'
        return MotionReport(reason,self._step-start,snapshot,distance)

    def set_gripper(self, arm, close, *, max_steps, on_step=None):
        if arm not in ARMS: raise McpToolError('invalid_input', 'Unknown arm')
        snapshot = self.observe()
        if max_steps <= 0 or self._step >= self.info.episode_budget:
            return MotionReport(LIMIT,0,snapshot)
        if self.driver.ended: return MotionReport('ended',0,snapshot)
        self._commands[arm] = 'close' if close else 'open'
        snapshot = self._execute(target_action(snapshot,self._commands),on_step)
        return MotionReport('command_applied',1,snapshot)

    def control_arms(self, targets, *, max_steps, on_step=None):
        from .coordinated import control_arms
        return control_arms(self, targets, max_steps=max_steps, on_step=on_step)

    def success(self):
        return self._snapshot is not None and bool(self.driver.success)

    def episode_audit(self):
        if self._snapshot is not None and self._audit is None:
            self._audit = self.driver.finish()
        return self._audit

    def end_episode(self):
        self.driver.close()
        self._snapshot = None
