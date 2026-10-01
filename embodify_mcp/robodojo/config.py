"""Pinned RoboDojo revisions and operator settings. Operator configuration is not an Agent tool surface."""
from __future__ import annotations
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from .tasks import TASKS

REVISION = '726e9aabfaa642203722eb126f5eaf0f37f3e1ad'
SUBMODULES = {'XPolicyLab': 'bb9a0b5f5136a74503b679af830bfd0a3a837d5c',
              'third_party/IsaacLab': 'afca7b09d60d8beb9c1cb28b43066499940b969b',
              'third_party/curobo': 'd17b54ce32cba095c0b000c4c58777075d11de0e'}
ASSET_REVISION = '43dacb13d3051a9ccd421f4e7e08e4e3eb29dfab'
DEFAULT_TASK = 'stack_blocks'
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
ARMS = ('left', 'right')


def validate_episode_budget(value):
    if type(value) is not int or value <= 0:
        raise ValueError('episode_budget must be a positive integer')
    return value


@dataclass(frozen=True)
class Settings:
    source: str
    python: str
    output: str
    #: Task a plain reset_task starts; the Agent may choose another unless --lock-task.
    task: str = DEFAULT_TASK
    seed: int = 0
    layout_index: int = 0
    startup_timeout_s: float = 600.0
    action_timeout_s: float = 180.0
    image_size: int = 256
    #: None keeps each task's native limit; an integer overrides it for every task.
    episode_budget: Optional[int] = None

    def __post_init__(self):
        if self.episode_budget is not None:
            validate_episode_budget(self.episode_budget)
        for field in ('source', 'python', 'output'):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError('missing ' + field)
        if not isinstance(self.task, str) or self.task not in TASKS:
            raise ValueError('unknown RoboDojo task')
        if type(self.seed) is not int or self.seed not in (0, 1, 2):
            raise ValueError('seed must be 0, 1 or 2')
        if type(self.layout_index) is not int or self.layout_index < 0:
            raise ValueError('layout_index must be nonnegative')
        if type(self.image_size) is not int or not 16 <= self.image_size <= 1024:
            raise ValueError('image_size must be 16..1024')
        for x in (self.startup_timeout_s, self.action_timeout_s):
            if isinstance(x, bool) or not math.isfinite(x) or x <= 0:
                raise ValueError('timeouts must be finite and positive')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        return cls(**data)
