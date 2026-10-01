"""Shared task identity types."""
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class TaskSpec:
    """A task-catalogue entry, independent of a simulator implementation."""

    suite: str
    task_index: int
    name: str
    instruction: str
    n_init_states: int = 0
    #: Actions per episode when the task sets its own limit; None means BackendInfo.episode_budget.
    episode_budget: Optional[int] = None
