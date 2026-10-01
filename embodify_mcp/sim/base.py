"""Abstract interface of the simulator adapter layer.

The real implementation (`libero_adapter.py`) imports libero / robosuite / mujoco; the fake one
(`fake.py`) imports no heavy dependencies. The layers above depend only on this interface, so the
whole chain can be tested on a laptop without a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..geometry import CameraModel
from ..obs import Observation


from ..types import TaskSpec  # Backwards-compatible internal import.


@dataclass(frozen=True)
class PrivilegedState:
    """Ground-truth state. **Only evaluation code and geometry self-checks may read it.**

    The keys of `object_positions` are the simulator's internal object names. The names are
    privileged too: they reveal what is in the scene, so they must never reach the agent verbatim.
    """

    object_positions: dict[str, np.ndarray] = field(default_factory=dict)
    object_quats: dict[str, np.ndarray] = field(default_factory=dict)
    target_object: str | None = None
    goal_positions: dict[str, np.ndarray] = field(default_factory=dict)


@runtime_checkable
class SimAdapter(Protocol):
    """The minimal interface a stepped backend needs."""

    @property
    def action_dim(self) -> int:
        """7 for LIBERO (3 position + 3 orientation + 1 gripper)."""

    @property
    def max_steps(self) -> int:
        """The current suite's horizon. Do not change it to suit a model, or published numbers stop being comparable."""

    def reset(self, task: TaskSpec, init_state_index: int) -> Observation: ...

    def step(self, action: np.ndarray) -> Observation: ...

    def observe(self) -> Observation: ...

    def privileged_state(self) -> PrivilegedState: ...

    def success(self) -> bool: ...

    def render(
        self, camera: str, width: int, height: int, *, depth: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]: ...

    def cameras(self) -> dict[str, CameraModel]: ...

    def close(self) -> None: ...


def check_action(action: np.ndarray, action_dim: int) -> np.ndarray:
    """One validity check for actions: correct dimension, finite, within [-1, 1].

    It lives here rather than in each adapter so that the fake and the real implementation share
    the same constraints; otherwise motions that work on the fake would be silently clipped on the
    real simulator.
    """
    arr = np.asarray(action, dtype=np.float64).reshape(-1)
    if arr.size != action_dim:
        raise ValueError(f"Action dimension should be {action_dim}, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Action contains non-finite values: {arr}")
    if np.any(np.abs(arr) > 1.0 + 1e-9):
        raise ValueError(f"Action outside [-1, 1]: {arr}")
    return arr
