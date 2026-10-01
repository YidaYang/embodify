"""A deterministic fake simulator: a table, a spherical object and a goal zone.

Its purpose is not realism but letting **all code except MuJoCo itself** be tested end to end on
machines without a simulator: the MCP session, the stepped control loop, back-projection and the
journal.

The kinematics are deliberately simplified (no dynamics, no contact forces), so:
- working here does not mean working on LIBERO;
- but failing here is always our own logic error, not a MuJoCo problem.

The synthetic cameras do real analytic ray casting (table plane + object spheres) and output
**normalized depth**, so back-projection follows exactly the same decoding path as the real adapter.
"""

from __future__ import annotations

import numpy as np

from ..geometry import (
    CameraModel,
    look_at_extrinsic,
    meters_to_opengl_depth,
    pinhole_intrinsic,
)
from ..obs import Observation
from ..primitives.controller import axis_angle_to_quat_xyzw, quat_xyzw_multiply
from .base import PrivilegedState, TaskSpec, check_action

TABLE_Z = 0.80
OBJECT_RADIUS = 0.025


class FakeSim:
    """A minimal kinematic simulator. Call `reset` after construction."""

    def __init__(
        self,
        *,
        action_dim: int = 7,
        max_steps: int = 250,
        max_pos_delta: float = 0.05,
        grasp_radius: float = 0.035,
        success_radius: float = 0.05,
        image_size: int = 128,
    ) -> None:
        self._action_dim = action_dim
        self._max_steps = max_steps
        self.max_pos_delta = max_pos_delta
        self.grasp_radius = grasp_radius
        self.success_radius = success_radius
        self.image_size = image_size
        self._task: TaskSpec | None = None
        self._closed = False
        self._eef_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.max_rotation_delta_rad = 0.5

    # -- interface properties ----------------------------------------------

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def max_steps(self) -> int:
        return self._max_steps

    # -- lifecycle ---------------------------------------------------------

    def reset(self, task: TaskSpec, init_state_index: int) -> Observation:
        """init_state_index fixes the object and goal layout, so the same index is fully reproducible."""
        rng = np.random.default_rng(abs(hash((task.suite, task.task_index, init_state_index))) % 2**32)
        self._task = task
        self._step_index = 0
        self._eef = np.array([0.0, 0.0, TABLE_Z + 0.25])
        self._eef_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self._gripper_closed = False
        self._holding = False
        self._object = np.array(
            [rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), TABLE_Z + OBJECT_RADIUS]
        )
        self._goal = np.array([rng.uniform(-0.15, 0.15), rng.uniform(0.20, 0.30), TABLE_Z])
        # A distractor, possibly of the same kind as the target: names alone cannot single out the target
        self._distractor = np.array(
            [rng.uniform(-0.15, 0.15), rng.uniform(-0.35, -0.25), TABLE_Z + OBJECT_RADIUS]
        )
        return self.observe()

    def close(self) -> None:
        self._closed = True

    # -- stepping ----------------------------------------------------------

    def step(self, action: np.ndarray) -> Observation:
        arr = check_action(action, self._action_dim)
        if self._task is None:
            raise RuntimeError("Reset before stepping")
        self._step_index += 1
        self._eef = self._eef + arr[:3] * self.max_pos_delta
        self._eef[2] = max(self._eef[2], TABLE_Z + 0.005)
        rotvec = arr[3:6] * self.max_rotation_delta_rad
        self._eef_quat = quat_xyzw_multiply(axis_angle_to_quat_xyzw(rotvec), self._eef_quat)

        want_closed = bool(arr[self._action_dim - 1] > 0.0)
        if want_closed and not self._gripper_closed:
            if float(np.linalg.norm(self._eef - self._object)) <= self.grasp_radius:
                self._holding = True
        if not want_closed and self._gripper_closed:
            self._holding = False
        self._gripper_closed = want_closed

        if self._holding:
            self._object = self._eef.copy()
        else:
            self._object[2] = TABLE_Z + OBJECT_RADIUS  # a released object drops onto the table
        return self.observe()

    def success(self) -> bool:
        """The object is in the goal zone and released."""
        if self._task is None or self._holding:
            return False
        return float(np.linalg.norm(self._object[:2] - self._goal[:2])) <= self.success_radius

    # -- observation -------------------------------------------------------

    def observe(self) -> Observation:
        if self._task is None:
            raise RuntimeError("Reset before observing")
        cams = self.cameras()
        rgb: dict[str, np.ndarray] = {}
        depth: dict[str, np.ndarray] = {}
        for name, cam in cams.items():
            rgb[name], depth[name] = self.render(name, cam.width, cam.height, depth=True)
        return Observation(
            instruction=self._task.instruction,
            step_index=self._step_index,
            rgb=rgb,
            depth=depth,
            cameras=cams,
            eef_pos=self._eef.copy(),
            eef_quat=self._eef_quat.copy(),
            gripper_width=0.0 if self._gripper_closed else 0.08,
        )

    def privileged_state(self) -> PrivilegedState:
        return PrivilegedState(
            object_positions={
                "target_object": self._object.copy(),
                "distractor_object": self._distractor.copy(),
            },
            object_quats={
                "target_object": np.array([0.0, 0.0, 0.0, 1.0]),
                "distractor_object": np.array([0.0, 0.0, 0.0, 1.0]),
            },
            target_object="target_object",
            goal_positions={"goal": self._goal.copy()},
        )

    def cameras(self) -> dict[str, CameraModel]:
        n = self.image_size
        return {
            "agentview": self._camera("agentview", n, eye=(0.0, -0.9, TABLE_Z + 0.65)),
            "eye_in_hand": self._camera("eye_in_hand", n, eye=tuple(self._eef + np.array([0.0, -0.06, 0.10]))),
        }

    def _camera(self, name: str, n: int, *, eye: tuple[float, float, float]) -> CameraModel:
        target = self._object if name == "eye_in_hand" else np.array([0.0, 0.0, TABLE_Z])
        return CameraModel(
            name=name,
            width=n,
            height=n,
            intrinsic=pinhole_intrinsic(n, n, fovy_deg=45.0),
            extrinsic=look_at_extrinsic(np.asarray(eye, dtype=np.float64), target, np.array([0.0, 0.0, -1.0])),
            near=0.05,
            far=5.0,
        )

    # -- synthetic rendering: analytic intersection with the table plane and spheres --

    def render(
        self, camera: str, width: int, height: int, *, depth: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        cams = self.cameras()
        if camera not in cams:
            raise KeyError(f"No camera {camera!r}")
        base = cams[camera]
        cam = CameraModel(
            name=base.name,
            width=width,
            height=height,
            intrinsic=pinhole_intrinsic(width, height, fovy_deg=45.0),
            extrinsic=base.extrinsic,
            near=base.near,
            far=base.far,
        )
        origin = cam.extrinsic[:3, 3]
        dirs = _ray_directions(cam)

        t = np.full((height, width), np.inf)
        color = np.zeros((height, width, 3), dtype=np.uint8)

        # Table: the plane z = TABLE_Z
        with np.errstate(divide="ignore", invalid="ignore"):
            t_plane = (TABLE_Z - origin[2]) / dirs[..., 2]
        hit_plane = np.isfinite(t_plane) & (t_plane > 0)
        t = np.where(hit_plane & (t_plane < t), t_plane, t)
        color[hit_plane] = (150, 150, 155)

        for center, rgb_value in (
            (self._object, (210, 70, 60)),
            (self._distractor, (70, 110, 200)),
            (self._goal + np.array([0.0, 0.0, 0.001]), (90, 170, 120)),
        ):
            radius = OBJECT_RADIUS if rgb_value != (90, 170, 120) else self.success_radius
            t_s = _sphere_hit(origin, dirs, np.asarray(center, dtype=np.float64), radius)
            closer = np.isfinite(t_s) & (t_s < t)
            t = np.where(closer, t_s, t)
            color[closer] = rgb_value

        if not depth:
            return color, None
        # t is the distance along the ray; depth is its component along the optical axis, as in the pinhole model
        axis_z = t * _axis_component(cam, dirs)
        meters = np.where(np.isfinite(axis_z), axis_z, cam.far)
        return color, meters_to_opengl_depth(meters, cam.near, cam.far)


def _ray_directions(cam: CameraModel) -> np.ndarray:
    """Unit direction in the world frame through each pixel center."""
    us, vs = np.meshgrid(np.arange(cam.width), np.arange(cam.height))
    fx, fy = cam.intrinsic[0, 0], cam.intrinsic[1, 1]
    cx, cy = cam.intrinsic[0, 2], cam.intrinsic[1, 2]
    x = (us - cx) / fx
    y = (vs - cy) / fy
    local = np.stack([x, y, np.ones_like(x)], axis=-1)
    local /= np.linalg.norm(local, axis=-1, keepdims=True)
    return local @ cam.extrinsic[:3, :3].T


def _axis_component(cam: CameraModel, dirs: np.ndarray) -> np.ndarray:
    """Projection of ray directions onto the optical axis, to turn distance into depth."""
    return dirs @ cam.extrinsic[:3, 2]


def _sphere_hit(
    origin: np.ndarray, dirs: np.ndarray, center: np.ndarray, radius: float
) -> np.ndarray:
    oc = origin - center
    b = 2.0 * (dirs @ oc)
    c = float(oc @ oc) - radius * radius
    disc = b * b - 4.0 * c
    out = np.full(dirs.shape[:2], np.inf)
    ok = disc >= 0.0
    sqrt_disc = np.sqrt(np.where(ok, disc, 0.0))
    t0 = (-b - sqrt_disc) / 2.0
    t1 = (-b + sqrt_disc) / 2.0
    t = np.where(t0 > 0, t0, t1)
    valid = ok & (t > 0)
    return np.where(valid, t, out)


def fake_tasks(n: int = 4, suite: str = "fake_suite") -> list[TaskSpec]:
    """A small set of tasks for tests."""
    return [
        TaskSpec(
            suite=suite,
            task_index=i,
            name=f"fake_task_{i}",
            instruction=f"pick up the red block and put it in the green zone ({i})",
            n_init_states=3,
        )
        for i in range(n)
    ]
