"""A three-level task catalogue: suite -> task -> initial layout. Shared by LIBERO and FakeSim.

The logic comes unchanged from the pre-refactor `LiberoMcpSession` (`_catalogue` /
`_select_scene` / `list_tasks`); only the LIBERO-specific texts became constructor arguments.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..sim.base import TaskSpec
from .base import Catalogue, CatalogueTexts, McpToolError

TaskSource = Callable[[str], Sequence[TaskSpec]]
SuiteSource = Callable[[], Sequence[Dict[str, Any]]]

#: LIBERO task names look like `KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet`: a scene
#: prefix plus the instruction with underscores. The prefix is information the instruction lacks
#: (the same "close the drawer" appears in several scenes); the rest repeats it word for word.
_SCENE_PREFIX = re.compile(r"^([A-Z][A-Z0-9_]*SCENE\d+)_(.+)$")


def _slug(text: str) -> str:
    """Squash a sentence into a comparable form, to tell whether a task name just repeats the instruction."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def task_row(suite: str, item: TaskSpec) -> Dict[str, Any]:
    """One row of `list_tasks`.

    Carries only what the Agent can use. In libero_spatial / libero_object / libero_goal the
    `name` equals the `instruction` word for word (spaces replaced by underscores), and
    `reset_task` takes the `task_index`, not the name, so including it would double every row.
    Therefore:
      * a name derivable from the instruction -> no `name`;
      * a name with a scene prefix -> only the prefix itself (`scene`), which is the new information;
      * a name that says something else -> `name` as is; better too much than lost information.
    """
    row: Dict[str, Any] = {
        "suite": suite,
        "task_index": int(item.task_index),
        "instruction": item.instruction,
        "n_init_states": int(item.n_init_states),
    }
    tail = item.name
    match = _SCENE_PREFIX.match(item.name)
    if match:
        row["scene"] = match.group(1)
        tail = match.group(2)
    if _slug(tail) != _slug(item.instruction):
        row["name"] = item.name
    return row


#: Of the three scene parameters, the two besides suite read the same for every suite-style catalogue.
TASK_INDEX_PARAM = {
    "type": "integer",
    "minimum": 0,
    "description": "Which task of the suite; omit to use the server's default scene (task 0 when switching suites). See list_tasks with suite for the options.",
}
INIT_STATE_PARAM = {
    "type": "integer",
    "minimum": 0,
    "description": "Which initial layout of the task; 0 if omitted. Must be below the task's n_init_states.",
}
RESET_INTRO = (
    "**All three arguments are optional**: omit them all to use the server's default scene "
    "(usually task 0 of the suite with initial layout 0), which is the most common use.\n"
)
RESET_HOWTO = (
    "To switch tasks, use list_tasks first: without arguments it lists the suites, with suite it gives "
    "the task_index and instruction of every task in that suite; then pass the chosen suite / task_index. "
    "init_state_index picks another object layout of the same task. "
    "If the server was started with --lock-task (task_locked is true in get_session_info), passing any of "
    "these arguments returns a task_locked error.\n"
)
TASK_FIELDS = "suite, task_index, name, instruction, init_state_index"


def suite_texts(*, reset_blurb: str, suite_param: str, list_description: str, list_suite_param: str) -> CatalogueTexts:
    """Texts for a suite-style catalogue: only the sentences introducing the benchmark differ between backends."""
    return CatalogueTexts(
        reset_intro=RESET_INTRO,
        reset_blurb=reset_blurb,
        reset_properties={
            "task_index": dict(TASK_INDEX_PARAM),
            "init_state_index": dict(INIT_STATE_PARAM),
            "suite": {"type": "string", "description": suite_param},
        },
        reset_howto=RESET_HOWTO,
        task_fields=TASK_FIELDS,
        list_description=list_description,
        list_properties={"suite": {"type": "string", "description": list_suite_param}},
    )


class SuiteCatalogue(Catalogue):
    """suite -> task -> initial layout.

    - `tasks`: the default suite's task list (optional; if omitted, `task_source` is asked on first use).
    - `task_source`: loads other suites on demand; without it the server offers only the default suite.
    - `suite_source`: reports every available suite and its task count; probing instantiates every
      benchmark, so it runs only once.
    """

    def __init__(
        self,
        *,
        texts: CatalogueTexts,
        default_suite: str,
        default_task_index: int = 0,
        default_init_state: int = 0,
        tasks: Optional[Sequence[TaskSpec]] = None,
        task_source: Optional[TaskSource] = None,
        suite_source: Optional[SuiteSource] = None,
    ) -> None:
        self._texts = texts
        self._task_source = task_source
        self._suite_source = suite_source
        self._suites: Optional[List[Dict[str, Any]]] = None
        self._catalogues: Dict[str, List[TaskSpec]] = {}
        if tasks:
            self._catalogues[default_suite] = list(tasks)
        # The "current" task: a reset_task without arguments lands here. Task lists load only when
        # actually needed, so the catalogue and tool descriptions work on machines without LIBERO.
        self._suite = default_suite
        self._index = int(default_task_index)
        self._init = int(default_init_state)
        self._task: Optional[TaskSpec] = None

    # -- Catalogue interface -----------------------------------------------

    def texts(self) -> CatalogueTexts:
        return self._texts

    def resolve(self, args: Dict[str, Any], *, locked: bool) -> Tuple[TaskSpec, int]:
        """Pick the scene from the arguments; with none, keep the current (initially the server default) scene."""
        wants = any(key in args for key in ("suite", "task_index", "init_state_index"))
        if wants and locked:
            raise McpToolError(
                "task_locked",
                "This run's scene is fixed (the server was started with --lock-task), so suite / task_index /"
                " init_state_index are not accepted. Call reset_task without arguments.",
            )
        if not wants:
            return self._current(), self._init

        suite = args.get("suite", self._suite)
        if not isinstance(suite, str):
            raise McpToolError("invalid_input", "suite must be a string")
        catalogue = self._catalogue(suite)

        index = args.get("task_index", 0 if suite != self._suite else self._index)
        if isinstance(index, bool) or not isinstance(index, int):
            raise McpToolError("invalid_input", "task_index must be an integer")
        if not 0 <= index < len(catalogue):
            raise McpToolError(
                "invalid_input",
                f"Suite {suite} has only {len(catalogue)} tasks (task_index 0..{len(catalogue) - 1}), "
                f"got {index}. See list_tasks for the available tasks.",
            )
        task = catalogue[index]

        init_index = args.get("init_state_index", 0)
        if isinstance(init_index, bool) or not isinstance(init_index, int):
            raise McpToolError("invalid_input", "init_state_index must be an integer")
        # n_init_states is 0 when the catalogue could not report it; only validate what we know.
        limit = int(task.n_init_states)
        if init_index < 0 or (limit > 0 and init_index >= limit):
            raise McpToolError(
                "invalid_input",
                f"{suite}/{task.name} has {limit} initial layouts (init_state_index 0..{limit - 1}), "
                f"got {init_index}.",
            )
        return task, init_index

    def commit(self, task: TaskSpec, variant: int) -> None:
        self._task = task
        self._suite = task.suite
        self._index = int(task.task_index)
        self._init = int(variant)

    def task_payload(self) -> Dict[str, Any]:
        task = self._current()
        return {
            "suite": task.suite,
            "task_index": int(task.task_index),
            "name": task.name,
            "instruction": task.instruction,
            "init_state_index": int(self._init),
        }

    def idle_payload(self) -> Dict[str, Any]:
        return {
            "suite": self._suite,
            "default_scene": {
                "suite": self._suite,
                "task_index": self._index,
                "init_state_index": self._init,
            },
        }

    def listing(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Two levels: without arguments only the suites, with suite that suite's tasks.

        There are two levels because all 130 LIBERO instructions together are about 27 KB. Picking a
        suite first means the Agent usually reads a tenth of it, and sees how large the whole is
        before asking for all of it.
        """
        wanted = args.get("suite")
        if wanted is None:
            suites = self._suite_catalogue()
            return {
                "ok": True,
                "suites": suites,
                "n_tasks_total": sum(int(item["n_tasks"]) for item in suites),
            }
        if not isinstance(wanted, str):
            raise McpToolError("invalid_input", "suite must be a string")
        return {"ok": True, "suite": wanted, "tasks": [task_row(wanted, item) for item in self._catalogue(wanted)]}

    # -- internals ---------------------------------------------------------

    def _current(self) -> TaskSpec:
        if self._task is None:
            catalogue = self._catalogue(self._suite)
            if not 0 <= self._index < len(catalogue):
                raise McpToolError(
                    "invalid_input",
                    f"The server's default scene {self._suite} #{self._index} does not exist"
                    f" (the suite has {len(catalogue)} tasks).",
                )
            self._task = catalogue[self._index]
        return self._task

    def _catalogue(self, suite: str) -> List[TaskSpec]:
        """The tasks of one suite, loaded on demand and cached."""
        if suite in self._catalogues:
            return self._catalogues[suite]
        if self._task_source is None:
            known = ", ".join(sorted(self._catalogues))
            raise McpToolError("invalid_input", f"This server only offers suite {known}, got {suite!r}")
        # When the server can enumerate its suites, the advertised list *is* the selectable set.
        # Letting the two drift apart is what makes "listed but unstartable" bugs possible.
        if self._suite_source is not None:
            offered = [str(item["suite"]) for item in self._suite_catalogue()]
            if suite not in offered:
                raise McpToolError(
                    "invalid_input",
                    f"No suite named {suite!r}. Available suites: {', '.join(offered)}. "
                    "See list_tasks (without arguments).",
                )
        try:
            loaded = list(self._task_source(suite))
        except McpToolError:
            raise
        except Exception as exc:
            known = ", ".join(str(item["suite"]) for item in self._suite_catalogue())
            raise McpToolError(
                "invalid_input",
                f"Cannot load suite {suite!r}: {exc}. Available suites: {known}.",
            )
        if not loaded:
            raise McpToolError("invalid_input", f"Suite {suite!r} has no tasks")
        self._catalogues[suite] = loaded
        return loaded

    def _suite_catalogue(self) -> List[Dict[str, Any]]:
        """Every suite this server can open, with its task count."""
        if self._suites is not None:
            return self._suites
        if self._suite_source is None:
            if self._suite not in self._catalogues and self._task_source is not None:
                self._catalogue(self._suite)
            self._suites = [
                {"suite": name, "n_tasks": len(tasks)}
                for name, tasks in sorted(self._catalogues.items())
            ]
            return self._suites
        try:
            probed = [dict(item) for item in self._suite_source()]
        except Exception:  # a broken probe must not take down an otherwise usable session
            probed = [
                {"suite": name, "n_tasks": len(tasks)}
                for name, tasks in sorted(self._catalogues.items())
            ]
        self._suites = probed
        return self._suites
