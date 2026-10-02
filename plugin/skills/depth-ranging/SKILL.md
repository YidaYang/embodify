---
name: depth-ranging
description: (Placeholder — not implemented yet) Measure distances from depth images and camera calibration to turn pixels into 3D positions.
disable-model-invocation: true
---

# Depth ranging (placeholder)

**Status: planned.** The MCP tools currently return RGB images only. Depth
images and camera calibration are not exposed yet, although depth conversion
helpers already exist inside the package.

Planned scope:

- Return a depth image and camera intrinsics/extrinsics alongside RGB where the
  backend can provide them.
- Given a pixel (for example from `object-segmentation`), report its distance
  from the camera and its 3D position in the robot frame.
- Measure gaps and heights (clearance under a shelf, height of a stack) before acting.

Until this exists, estimate distances by comparing camera views and by probing
with small moves, then record what you measured in the robot profile.
