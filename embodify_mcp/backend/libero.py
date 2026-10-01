"""LIBERO backend: `LiberoAdapter` + the stepped loop + LIBERO's task catalogue and descriptions.

Everything LIBERO-specific (5 suites, 130 tasks, 50 initial layouts each, camera names, frame
wording, words that must never appear in results) lives in this file; the MCP layer hard-codes
none of it.

Constructing it does not need LIBERO installed: the simulator and the task lists load on first
use, so tool descriptions can be generated and tested on machines without the simulator.
"""

from __future__ import annotations

from typing import Optional

from .catalogue import SuiteCatalogue, suite_texts
from .stepped import SteppedSimBackend

#: Cameras: (public name, name in LIBERO observations, human-readable label).
LIBERO_CAMERAS = (
    ("agentview", "agentview", "Third person · agentview"),
    ("robot0_eye_in_hand", "robot0_eye_in_hand", "Wrist camera · eye in hand"),
)

#: Words that must not appear in any tool result (including descriptions): BDDL goals, success,
#: ground-truth poses, initial-state files.
LIBERO_FORBIDDEN = ("bddl", "success", "privileged", "object_pose", "goal_position", "init_state_file")

LIBERO_TEXTS = suite_texts(
    reset_blurb=(
        "**The whole LIBERO benchmark is available**: 130 tasks in 5 suites, each task with 50 initial "
        "layouts. libero_spatial, libero_object, libero_goal and libero_10 have 10 tasks each, libero_90 "
        "has 90. Passing only suite starts task 0 of that suite.\n"
    ),
    suite_param=(
        "Switch to another LIBERO suite; omit to keep the current one. All 5 are available: libero_spatial, "
        "libero_object, libero_goal, libero_10 (LIBERO-Long) and libero_90. See list_tasks (without arguments)."
    ),
    list_description=(
        "List the tasks LIBERO offers. **Independent of the current run**: callable at any time, renders no "
        "images, uses no step budget and does not change a running episode.\n"
        "Without arguments: returns suites, the name, task count and a one-line summary of all 5 suites, "
        "130 tasks in total. Start here, pick a suite, then look inside it.\n"
        "With suite: returns task_index, instruction and n_init_states (the number of initial layouts of the "
        "same task, always 50 in LIBERO) for every task in that suite. Tasks spread over several scenes also "
        "carry scene (a scene name such as KITCHEN_SCENE10), since the same instruction can appear in different "
        "scenes. A few tasks also have a name field, present only when the task name says more than the "
        "instruction.\n"
        "Pass the chosen suite and task_index to reset_task to start.\n"
        "There are two levels because all 130 instructions together are about 27 KB; picking a suite first "
        "usually means reading only a small part. To see everything, list the suites one by one."
    ),
    list_suite_param=(
        "The suite whose tasks to list; omit to get only the suite overview. One of libero_spatial, "
        "libero_object, libero_goal, libero_10, libero_90."
    ),
)


class LiberoBackend(SteppedSimBackend):
    def __init__(
        self,
        *,
        image_size: int = 256,
        max_steps: int = 1000,
        suite: str = "libero_spatial",
        task_index: int = 0,
        init_state_index: int = 0,
        renderer: Optional[str] = None,
        controller_verified: bool = False,
    ) -> None:
        from ..sim.libero_adapter import LiberoAdapter, list_suites, list_tasks

        adapter = LiberoAdapter(
            image_size=image_size,
            cameras=tuple(internal for _, internal, _ in LIBERO_CAMERAS),
            max_steps=max_steps,
            renderer=renderer,
        )
        catalogue = SuiteCatalogue(
            texts=LIBERO_TEXTS,
            default_suite=suite,
            default_task_index=task_index,
            default_init_state=init_state_index,
            task_source=list_tasks,
            suite_source=list_suites,
        )
        super().__init__(
            adapter,
            catalogue=catalogue,
            name="libero",
            title="LIBERO",
            server_name="libero-sim",
            frame="the LIBERO robot base frame",
            cameras=LIBERO_CAMERAS,
            forbidden_terms=LIBERO_FORBIDDEN,
            controller_verified=controller_verified,
        )
