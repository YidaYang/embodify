"""Simulator observation data; public MCP fields are selected by McpSession."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .geometry import CameraModel

@dataclass(frozen=True)
class Observation:
    """Everything the agent can see. The fields are the allowlist itself."""

    instruction: str
    step_index: int
    rgb: dict[str, np.ndarray]
    eef_pos: np.ndarray
    eef_quat: np.ndarray
    gripper_width: float
    depth: dict[str, np.ndarray] = field(default_factory=dict)
    cameras: dict[str, CameraModel] = field(default_factory=dict)

    def camera_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.rgb))

    def require_camera(self, name: str) -> CameraModel:
        if name not in self.cameras:
            raise KeyError(f"No calibration for camera {name!r}; available: {sorted(self.cameras)}")
        return self.cameras[name]
