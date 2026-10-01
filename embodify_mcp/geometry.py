"""Geometry: pixel <-> camera <-> world.

Back-projection lives here: a 2-D point in an image, together with depth and the camera
intrinsics and extrinsics, is lifted to world coordinates.

Two conventions must be checked against the simulator before these numbers are trusted on a new
backend:

1. **Meaning of the depth buffer.** robosuite's `camera_obs` gives OpenGL normalized depth by
   default ([0, 1], nonlinear), which `opengl_depth_to_meters` converts to meters. robosuite also
   ships `camera_utils.get_real_depth_map`; cross-check the two, and if they disagree, robosuite wins.
2. **Extrinsic direction.** `extrinsic` is assumed to be a 4x4 **camera-to-world** matrix
   (left-multiplying a point in camera coordinates gives world coordinates). robosuite documents
   `get_camera_extrinsic_matrix` as camera-to-world, but confirm the sign conventions (especially
   the z axis) with `roundtrip_error`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    """The full calibration of one camera.

    `intrinsic` is 3x3, `extrinsic` is 4x4 camera-to-world, and `near`/`far` are MuJoCo's
    clipping planes (used to convert normalized depth to meters).
    """

    name: str
    width: int
    height: int
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    near: float
    far: float

    def __post_init__(self) -> None:
        if self.intrinsic.shape != (3, 3):
            raise ValueError(f"intrinsic must be 3x3, got {self.intrinsic.shape}")
        if self.extrinsic.shape != (4, 4):
            raise ValueError(f"extrinsic must be 4x4, got {self.extrinsic.shape}")
        if not 0.0 < self.near < self.far:
            raise ValueError(f"Need 0 < near < far, got near={self.near} far={self.far}")


def opengl_depth_to_meters(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """Convert an OpenGL normalized depth buffer ([0, 1]) to meters.

    Standard inverse formula: `z = 2*near*far / (far + near - (2*d - 1)*(far - near))`.
    Cross-check it against robosuite's `get_real_depth_map` on a new setup.
    """
    d = np.asarray(depth, dtype=np.float64)
    denom = far + near - (2.0 * d - 1.0) * (far - near)
    with np.errstate(divide="ignore", invalid="ignore"):
        meters = (2.0 * near * far) / denom
    return np.where(np.isfinite(meters), meters, far)


def meters_to_opengl_depth(meters: np.ndarray, near: float, far: float) -> np.ndarray:
    """The inverse of `opengl_depth_to_meters`.

    It lets the synthetic cameras (`sim/fake.py`) output normalized depth too, so unit tests follow
    **exactly the same** decoding path as the real adapter, and a depth transform cannot pass on the
    fake yet be wrong on the real simulator.
    """
    z = np.clip(np.asarray(meters, dtype=np.float64), near, far)
    return 0.5 * (((far + near) - (2.0 * near * far) / z) / (far - near) + 1.0)



def pixel_to_camera(u: float, v: float, z: float, intrinsic: np.ndarray) -> np.ndarray:
    """Convert pixel (u, v) and depth z along the optical axis (meters) to a 3D point in the camera frame.

    Camera frame convention: x right, y down, z forward along the optical axis, the standard
    pinhole convention. If a simulator turns out to use the OpenGL convention (y up, z backward),
    add a fixed flip matrix in `camera_to_world` instead of changing this function.
    """
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    if fx == 0.0 or fy == 0.0:
        raise ValueError("Zero focal length in the intrinsics; the calibration was not read correctly")
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy
    return np.array([x, y, float(z)], dtype=np.float64)


def camera_to_world(point_cam: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """Camera coordinates -> world coordinates; `extrinsic` is 4x4 camera-to-world."""
    p = np.asarray(point_cam, dtype=np.float64).reshape(3)
    homo = np.concatenate([p, [1.0]])
    return (np.asarray(extrinsic, dtype=np.float64) @ homo)[:3]


def world_to_camera(point_world: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """World coordinates -> camera coordinates (the inverse of camera-to-world)."""
    p = np.asarray(point_world, dtype=np.float64).reshape(3)
    homo = np.concatenate([p, [1.0]])
    inv = np.linalg.inv(np.asarray(extrinsic, dtype=np.float64))
    return (inv @ homo)[:3]


def project(point_world: np.ndarray, cam: CameraModel) -> tuple[float, float]:
    """World coordinates -> pixel (u, v). Raises for points behind the camera instead of silently returning garbage."""
    p_cam = world_to_camera(point_world, cam.extrinsic)
    if p_cam[2] <= 0.0:
        raise ValueError(f"The point is behind the camera (z={p_cam[2]:.4f}) and cannot be projected")
    u = cam.intrinsic[0, 0] * p_cam[0] / p_cam[2] + cam.intrinsic[0, 2]
    v = cam.intrinsic[1, 1] * p_cam[1] / p_cam[2] + cam.intrinsic[1, 2]
    return float(u), float(v)


def unproject_pixel(
    u: float,
    v: float,
    depth_map: np.ndarray,
    cam: CameraModel,
    *,
    depth_is_normalized: bool = True,
    patch: int = 1,
) -> np.ndarray:
    """Lift one image point to world coordinates.

    With `patch>1` the median of the valid depths in the (2*patch+1)^2 neighborhood is used:
    picked points often land on object edges, where a single pixel's depth hits the background,
    and the median cheaply guards against that.
    """
    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth_map must be 2D, got {depth.shape}")
    if depth_is_normalized:
        depth = opengl_depth_to_meters(depth, cam.near, cam.far)

    ui = int(round(u))
    vi = int(round(v))
    if not (0 <= ui < depth.shape[1] and 0 <= vi < depth.shape[0]):
        raise ValueError(f"Pixel ({ui},{vi}) is outside the {depth.shape[1]}x{depth.shape[0]} image")

    lo_v, hi_v = max(0, vi - patch), min(depth.shape[0], vi + patch + 1)
    lo_u, hi_u = max(0, ui - patch), min(depth.shape[1], ui + patch + 1)
    window = depth[lo_v:hi_v, lo_u:hi_u]
    valid = window[(window > cam.near) & (window < cam.far)]
    if valid.size == 0:
        raise ValueError(f"No valid depth around pixel ({ui},{vi})")
    z = float(np.median(valid))
    return camera_to_world(pixel_to_camera(ui, vi, z, cam.intrinsic), cam.extrinsic)


def roundtrip_error(point_world: np.ndarray, cam: CameraModel) -> float:
    """Self-check: world point -> projection -> back-projection with the analytic depth -> error in meters.

    It uses **geometry only** and needs no simulator. Running the same round trip with
    ground-truth object poses from a simulator turns "projection then back-projection < 1 mm" into
    a measurement; ground-truth poses are used only for this self-check and never reach the agent.
    """
    u, v = project(point_world, cam)
    z = float(world_to_camera(point_world, cam.extrinsic)[2])
    back = camera_to_world(pixel_to_camera(u, v, z, cam.intrinsic), cam.extrinsic)
    return float(np.linalg.norm(back - np.asarray(point_world, dtype=np.float64).reshape(3)))


def pinhole_intrinsic(width: int, height: int, fovy_deg: float) -> np.ndarray:
    """Build pinhole intrinsics from MuJoCo's vertical FOV. For tests and synthetic cameras only."""
    f = 0.5 * height / np.tan(0.5 * np.deg2rad(fovy_deg))
    return np.array(
        [[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def look_at_extrinsic(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build a camera-to-world extrinsic (x right, y down, z forward). For tests and synthetic cameras only."""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    forward = np.asarray(target, dtype=np.float64).reshape(3) - eye
    norm = np.linalg.norm(forward)
    if norm == 0.0:
        raise ValueError("eye and target coincide")
    forward /= norm
    right = np.cross(np.asarray(up, dtype=np.float64).reshape(3), forward)
    rnorm = np.linalg.norm(right)
    if rnorm < 1e-9:
        raise ValueError("up is parallel to the line of sight; cannot build an orthogonal basis")
    right /= rnorm
    down = np.cross(forward, right)
    ext = np.eye(4, dtype=np.float64)
    ext[:3, 0] = right
    ext[:3, 1] = down
    ext[:3, 2] = forward
    ext[:3, 3] = eye
    return ext
