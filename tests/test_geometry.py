"""Geometry: projection round trips, depth encoding and an end-to-end check on synthetic cameras."""

from __future__ import annotations

import unittest

import numpy as np

from embodify_mcp.geometry import (
    CameraModel,
    look_at_extrinsic,
    meters_to_opengl_depth,
    opengl_depth_to_meters,
    pinhole_intrinsic,
    project,
    roundtrip_error,
    unproject_pixel,
    world_to_camera,
)
from embodify_mcp.sim.fake import OBJECT_RADIUS, TABLE_Z, FakeSim, fake_tasks


def _camera(n: int = 128) -> CameraModel:
    return CameraModel(
        name="test",
        width=n,
        height=n,
        intrinsic=pinhole_intrinsic(n, n, 45.0),
        extrinsic=look_at_extrinsic(
            np.array([0.0, -0.8, TABLE_Z + 0.6]), np.array([0.0, 0.0, TABLE_Z]), np.array([0.0, 0.0, -1.0])
        ),
        near=0.05,
        far=5.0,
    )


class GeometryTest(unittest.TestCase):
    def test_depth_encoding_roundtrip(self) -> None:
        meters = np.linspace(0.1, 4.0, 25)
        encoded = meters_to_opengl_depth(meters, 0.05, 5.0)
        decoded = opengl_depth_to_meters(encoded, 0.05, 5.0)
        self.assertTrue(np.all(encoded >= 0.0) and np.all(encoded <= 1.0))
        np.testing.assert_allclose(decoded, meters, rtol=1e-9, atol=1e-9)

    def test_projection_roundtrip_is_sub_millimetre(self) -> None:
        cam = _camera()
        for point in ([0.0, 0.0, TABLE_Z], [0.08, -0.05, TABLE_Z + 0.03], [-0.1, 0.12, TABLE_Z + 0.01]):
            self.assertLess(roundtrip_error(np.array(point), cam), 1e-3)

    def test_point_behind_camera_raises(self) -> None:
        cam = _camera()
        behind = cam.extrinsic[:3, 3] - cam.extrinsic[:3, 2] * 0.5
        with self.assertRaises(ValueError):
            project(behind, cam)


class UnprojectOnFakeSimTest(unittest.TestCase):
    """Pixel to world on synthetic RGBD, the basis of back-projection."""

    def setUp(self) -> None:
        self.sim = FakeSim(image_size=192)
        self.task = fake_tasks(1)[0]
        self.sim.reset(self.task, 0)

    def test_unproject_recovers_object_position(self) -> None:
        obs = self.sim.observe()
        state = self.sim.privileged_state()
        target = state.object_positions["target_object"]
        cam = obs.cameras["agentview"]
        u, v = project(target, cam)

        recovered = unproject_pixel(u, v, obs.depth["agentview"], cam, patch=1)
        # Back-projection lands on the sphere's surface, not its center, so the criterion is "within
        # the radius", not "equal to the center"; the lateral error is of order r*cos(elevation).
        error = float(np.linalg.norm(recovered - target))
        self.assertLess(
            error,
            OBJECT_RADIUS + 0.004,
            f"3D error {error:.4f} m exceeds the sphere radius: camera parameters or depth conventions disagree",
        )

    def test_unproject_rejects_out_of_bounds(self) -> None:
        obs = self.sim.observe()
        cam = obs.cameras["agentview"]
        with self.assertRaises(ValueError):
            unproject_pixel(-5, 10, obs.depth["agentview"], cam)

    def test_depth_matches_analytic_distance_on_table(self) -> None:
        """A table point that no object occludes must back-project onto the table.

        x=0.25 is used: the target object is sampled within |x|, |y| <= 0.15, the goal zone lies in
        y in [0.20, 0.30] and the distractor in y in [-0.35, -0.25], so this point is always bare table.
        """
        obs = self.sim.observe()
        cam = obs.cameras["agentview"]
        probe = np.array([0.25, 0.0, TABLE_Z])
        u, v = project(probe, cam)
        recovered = unproject_pixel(u, v, obs.depth["agentview"], cam, patch=0)
        self.assertGreater(float(world_to_camera(probe, cam.extrinsic)[2]), 0.0)
        self.assertLess(float(np.linalg.norm(recovered - probe)), 0.01)


if __name__ == "__main__":
    unittest.main()
