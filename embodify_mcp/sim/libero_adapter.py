"""Lazy LIBERO adapter; see docs/backends.md for runtime requirements.

Controller and camera assumptions must be verified against the installed
upstream runtime. Importing this module does not initialize a simulator.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..errors import SimNotAvailableError
from ..geometry import CameraModel
from ..obs import Observation
from .base import PrivilegedState, TaskSpec, check_action

#: The two cameras of standard LIBERO evaluation.
DEFAULT_CAMERAS = ("agentview", "robot0_eye_in_hand")

#: Official suite names; LIBERO-Long is libero_10 in the code. The benchmark registry also has
#: `libero_100`, but it has no BDDL directory and raises `KeyError: 'libero_100'` when
#: instantiated, so it is not listed here; `list_suites()` also drops suites that fail to load.
SUITE_NAMES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


def _lazy() -> tuple[Any, Any, Any]:
    """Import libero / robosuite lazily, with an actionable error on failure."""
    try:
        from libero.libero import benchmark as libero_benchmark  # type: ignore
        from libero.libero import get_libero_path  # type: ignore
        from libero.libero.envs import OffScreenRenderEnv  # type: ignore
    except ImportError as exc:  # pragma: no cover - no simulator on this machine
        raise SimNotAvailableError(
            "LIBERO is not installed. Pure-logic tests do not need it; to run episodes, install it"
            " on Linux or WSL2 as described in docs/backends.md and set MUJOCO_GL=osmesa (no GPU) or"
            f" egl (GPU). Original error: {exc}"
        ) from exc
    return libero_benchmark, get_libero_path, OffScreenRenderEnv


#: A one-line summary per suite, **measured from the task data, not written from memory**: the 10
#: libero_spatial instructions all put the black bowl on the plate and vary only the position
#: description; the 10 libero_object instructions are all "pick up X and place it in the basket"
#: with different objects; libero_goal has the shortest instructions (7.2 words on average) and
#: varied actions; libero_10 spans 9 scenes, 9 of 10 instructions contain "and", 13.2 words on
#: average; the 90 libero_90 tasks cover 20 scenes. With one line per suite, the Agent need not
#: pull the whole list to know which suite to look at.
SUITE_SUMMARIES: dict[str, str] = {
    "libero_spatial": "Same objects and goal (put the black bowl on the plate); only the position of the bowl to pick changes. Tests spatial reference.",
    "libero_object": "Same action (put an object in the basket) with different objects. Tests object recognition.",
    "libero_goal": "Same scene with different goals; the shortest instructions and the most varied actions (open a drawer, place objects, turn on the stove).",
    "libero_10": "Long-horizon tasks (LIBERO-Long) across 9 scenes; most need two or more steps.",
    "libero_90": "90 short-horizon tasks covering 20 scenes, the most varied suite here.",
}


def list_suites() -> list[dict[str, object]]:
    """List the suites that actually load on this machine, with their task counts.

    Reads only benchmark metadata and builds no environment, so it is fast (tens of milliseconds).
    Instantiating each suite is necessary: `get_benchmark_dict()` has keys that are registered but
    cannot load (`libero_100`), and reporting raw keys would let the Agent pick a suite that always fails.
    """
    libero_benchmark, _, _ = _lazy()
    registry = libero_benchmark.get_benchmark_dict()
    out: list[dict[str, object]] = []
    for name in SUITE_NAMES:
        cls = registry.get(name)
        if cls is None:
            continue
        try:
            bench = cls()
        except Exception:  # a registered suite that cannot load must not reach the Agent's list
            continue
        out.append(
            {
                "suite": name,
                "n_tasks": int(bench.n_tasks),
                "summary": SUITE_SUMMARIES.get(name, ""),
            }
        )
    return out


def list_tasks(suite: str) -> list[TaskSpec]:
    """List the tasks of one suite. Loads no environment, so it is fast."""
    libero_benchmark, _, _ = _lazy()
    suite_cls = libero_benchmark.get_benchmark_dict()[suite]
    bench = suite_cls()
    out: list[TaskSpec] = []
    for i in range(bench.n_tasks):
        task = bench.get_task(i)
        out.append(
            TaskSpec(
                suite=suite,
                task_index=i,
                name=getattr(task, "name", f"{suite}_{i}"),
                instruction=getattr(task, "language", ""),
                n_init_states=len(bench.get_task_init_states(i)),
            )
        )
    return out


class LiberoAdapter:
    """Wraps LIBERO's `OffScreenRenderEnv` as a `SimAdapter`."""

    def __init__(
        self,
        *,
        image_size: int = 128,
        cameras: tuple[str, ...] = DEFAULT_CAMERAS,
        max_steps: int = 250,
        renderer: str | None = None,
    ) -> None:
        # Module imports are not a sandbox. robosuite may legitimately import h5py.
        if renderer:
            os.environ.setdefault("MUJOCO_GL", renderer)
        self.image_size = image_size
        self.camera_names = tuple(cameras)
        self._max_steps = max_steps
        self._env: Any | None = None
        self._bench: Any | None = None
        self._task: TaskSpec | None = None
        self._step_index = 0
        self._last_obs: dict[str, Any] | None = None
        self._done = False

    # -- interface properties ----------------------------------------------

    @property
    def action_dim(self) -> int:
        """LIBERO uses 7-D OSC_POSE delta actions."""
        if self._env is not None:
            return int(self._env.env.action_dim)
        return 7

    @property
    def max_steps(self) -> int:
        """The horizon differs between suites (LIBERO-Long is clearly longer).

        **Do not change it to suit a model**: published results would no longer be comparable.
        """
        return self._max_steps

    # -- lifecycle ---------------------------------------------------------

    def reset(self, task: TaskSpec, init_state_index: int) -> Observation:
        libero_benchmark, get_libero_path, OffScreenRenderEnv = _lazy()
        if self._bench is None or self._task is None or self._task.suite != task.suite:
            self._bench = libero_benchmark.get_benchmark_dict()[task.suite]()
        bench = self._bench
        bddl_dir = get_libero_path("bddl_files")
        libero_task = bench.get_task(task.task_index)
        bddl_path = os.path.join(bddl_dir, libero_task.problem_folder, libero_task.bddl_file)

        if self._env is not None:
            self._env.close()
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=self.image_size,
            camera_widths=self.image_size,
            camera_names=list(self.camera_names),
            camera_depths=True,  # depth is needed for back-projection
        )
        self._env.seed(init_state_index)
        self._last_obs = self._env.reset()
        init_states = bench.get_task_init_states(task.task_index)
        if init_state_index >= len(init_states):
            raise IndexError(
                f"{task.suite}/{task.name} has only {len(init_states)} init states,"
                f" requested index {init_state_index}"
            )
        self._last_obs = self._env.set_init_state(init_states[init_state_index])
        self._task = task
        self._step_index = 0
        self._done = False
        return self.observe()

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    # -- stepping ----------------------------------------------------------

    def step(self, action: np.ndarray) -> Observation:
        if self._env is None:
            raise RuntimeError("Reset before stepping")
        arr = check_action(action, self.action_dim)
        obs, _reward, done, _info = self._env.step(arr.tolist())
        self._last_obs = obs
        self._done = bool(done)
        self._step_index += 1
        return self.observe()

    def success(self) -> bool:
        """LIBERO's success comes from BDDL predicates; `done` is true exactly on success.

        If `done` ever turns true on a timeout too, call `env.env._check_success()` directly instead.
        """
        return self._done

    # -- observation -------------------------------------------------------

    def observe(self) -> Observation:
        if self._last_obs is None or self._task is None:
            raise RuntimeError("Reset before observing")
        raw = self._last_obs
        rgb: dict[str, np.ndarray] = {}
        depth: dict[str, np.ndarray] = {}
        for cam in self.camera_names:
            # LIBERO/robosuite images are upside down, hence [::-1]
            image = raw.get(f"{cam}_image")
            if image is not None:
                rgb[cam] = np.asarray(image)[::-1]
            dmap = raw.get(f"{cam}_depth")
            if dmap is not None:
                depth[cam] = np.asarray(dmap)[::-1].squeeze()
        return Observation(
            instruction=self._task.instruction,
            step_index=self._step_index,
            rgb=rgb,
            depth=depth,
            cameras=self.cameras(),
            eef_pos=np.asarray(raw["robot0_eef_pos"], dtype=np.float64),
            eef_quat=np.asarray(raw["robot0_eef_quat"], dtype=np.float64),
            gripper_width=float(np.sum(np.abs(raw.get("robot0_gripper_qpos", [0.0])))),
        )

    def cameras(self) -> dict[str, CameraModel]:
        """Read intrinsics and extrinsics from robosuite.

        Three things to confirm on a new setup: whether `get_camera_extrinsic_matrix` is
        camera-to-world; whether near/far follow from `sim.model.vis.map` or `stat.extent`; and
        whether the camera frame uses the pinhole or the OpenGL convention (the latter has y up and
        z backward and needs diag(1, -1, -1) on the extrinsic).
        `embodify_mcp.geometry.roundtrip_error` settles all three.
        """
        if self._env is None:
            raise RuntimeError("Reset before reading the camera calibration")
        from robosuite.utils import camera_utils  # type: ignore

        sim = self._env.env.sim
        out: dict[str, CameraModel] = {}
        near, far = _clip_planes(sim)
        for cam in self.camera_names:
            intr = camera_utils.get_camera_intrinsic_matrix(
                sim=sim, camera_name=cam, camera_height=self.image_size, camera_width=self.image_size
            )
            extr = camera_utils.get_camera_extrinsic_matrix(sim=sim, camera_name=cam)
            out[cam] = CameraModel(
                name=cam,
                width=self.image_size,
                height=self.image_size,
                intrinsic=np.asarray(intr, dtype=np.float64),
                extrinsic=np.asarray(extr, dtype=np.float64),
                near=near,
                far=far,
            )
        return out

    def render(
        self, camera: str, width: int, height: int, *, depth: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Re-render the current state at a given resolution.

        The physical state does not change, so one state can be rendered at 128, 256 or 512 for free.
        """
        if self._env is None:
            raise RuntimeError("Reset before rendering")
        sim = self._env.env.sim
        image = sim.render(width=width, height=height, camera_name=camera, depth=depth)
        if depth:
            rgb, dmap = image
            return np.asarray(rgb)[::-1], np.asarray(dmap)[::-1]
        return np.asarray(image)[::-1], None

    # -- privileged channel (evaluation and post-hoc analysis only) ----------

    def privileged_state(self) -> PrivilegedState:
        """Ground-truth object poses.

        Goal regions (`goal_positions`) are not derived: LIBERO states goals as BDDL predicates
        (`On`, `In` and the like), and turning them into a world position needs predicate solving.
        An empty dict is returned instead of a made-up coordinate.
        """
        if self._env is None:
            raise RuntimeError("Reset before reading the ground-truth state")
        env = self._env.env
        positions: dict[str, np.ndarray] = {}
        quats: dict[str, np.ndarray] = {}
        for name, obj in getattr(env, "objects_dict", {}).items():
            try:
                body_id = env.sim.model.body_name2id(obj.root_body)
                positions[name] = np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64)
                quats[name] = np.asarray(env.sim.data.body_xquat[body_id], dtype=np.float64)
            except (KeyError, ValueError):
                continue
        return PrivilegedState(
            object_positions=positions,
            object_quats=quats,
            target_object=None,  # not derived from the BDDL predicates
            goal_positions={},  # not derived from the BDDL predicates
        )


def _clip_planes(sim: Any) -> tuple[float, float]:
    """MuJoCo's near/far clipping planes; znear/zfar are fractions of `stat.extent`."""
    extent = float(sim.model.stat.extent)
    near = float(sim.model.vis.map.znear) * extent
    far = float(sim.model.vis.map.zfar) * extent
    return near, far
