---
name: object-segmentation
description: (Placeholder — not implemented yet) Segment and locate objects in robot camera images with a vision model.
disable-model-invocation: true
---

# Object segmentation (placeholder)

**Status: planned.** Embodify does not ship a segmentation tool yet.

Planned scope:

- Take a camera image returned by `observe` and a text query ("the red mug").
- Return masks, bounding boxes and pixel centroids from an open-vocabulary
  segmentation model running on the user's machine or a remote GPU.
- Combined with depth and camera calibration (see `depth-ranging`), convert a
  centroid into a 3D target in the robot frame.

Until this exists, locate objects by comparing camera views and approaching
with small moves, as described in the `embodied-control` skill.
