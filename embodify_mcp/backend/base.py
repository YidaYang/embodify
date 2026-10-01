"""Interface between the MCP session and a backend, cut at "one action", not "one simulation step".

The MCP session (`embodify_mcp/mcp.py`) only owns the episode lifecycle, the step budget,
logging and the information boundary; how to bring an end effector to a target pose is the
backend's job. The cut sits here because all three kinds of backend require it:

- stepped simulators (LIBERO, FakeSim): the P-control loop lives in `stepped.py`, on the
  backend side;
- real robots (SO-101): the servo loop has to run next to the robot, not through MCP on
  every step;
- remote backends / RoboDojo: one action can take dozens of steps, a network round trip per
  step is unacceptable, and RoboDojo's own evaluation loop sets the pace.

A backend **never faces the Agent directly**: it returns raw poses, images and evaluation
signals, and the MCP session decides what the Agent may see and in what shape. `success()`
is for evaluation only; the session never puts it into any tool result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..types import TaskSpec


class McpToolError(Exception):
    """An error that can be shown to the Agent as is (no Python traceback).

    It lives at the backend layer because the task catalogue (`catalogue.py`) raises it too
    when it parses the Agent's choice.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class BackendDisconnected(McpToolError):
    """The connection or protocol failed and the old episode cannot continue; MCP must record the abort."""


@dataclass(frozen=True)
class CameraInfo:
    """One camera. `name` is the public name: the Agent sees it in images[].camera, and archived frames are named after it."""

    name: str
    #: Human-readable label for the monitor page, such as "Third person · agentview".
    label: str = ""


@dataclass(frozen=True)
class ArmInfo:
    """One arm."""

    name: str
    #: Base axes whose rotation this arm can control. A 6-DoF arm has ("x", "y", "z");
    #: the 5-DoF SO-101 lacks one; an empty tuple means position control only.
    rotation_axes: Tuple[str, ...] = ("x", "y", "z")
    has_gripper: bool = True


@dataclass(frozen=True)
class BackendInfo:
    """The backend's self-description. Tool descriptions, the monitor page and logs all read it, so the MCP layer hard-codes nothing."""

    #: Machine-readable name recorded in the journal manifest (such as "libero" or "fake").
    name: str
    #: Name shown to the Agent in tool descriptions (such as "LIBERO").
    title: str
    #: serverInfo.name reported at MCP initialize.
    server_name: str
    #: The frame move_relative uses, as a short phrase (such as "the LIBERO robot base frame").
    frame: str
    arms: Tuple[ArmInfo, ...]
    cameras: Tuple[CameraInfo, ...]
    #: Budget per episode, in budget_unit.
    episode_budget: int
    #: Budget unit as a plural noun for tool descriptions ("simulation steps" here; a real robot might use "seconds").
    budget_unit: str = "simulation steps"
    #: Whether the backend can judge task success (a real robot cannot; a person has to).
    has_success: bool = True
    #: Backend-specific words that must never appear in any tool result; the contract tests check them.
    forbidden_terms: Tuple[str, ...] = ()
    #: The sentence in the move_relative description on how the server executes a motion.
    motion_text: str = ""
    #: The sentence in the set_gripper description on how the server executes a gripper command.
    gripper_text: str = ""
    #: Move stop reasons the backend reports, with their meaning (the MCP session adds step_cap / budget).
    move_stop_reasons: Tuple[Tuple[str, str], ...] = ()
    #: Gripper stop reasons the backend reports, with their meaning (the session adds step_cap / budget / no_op).
    gripper_stop_reasons: Tuple[Tuple[str, str], ...] = ()
    #: Whether the gripper opening is measured in meters; otherwise snapshot openings are None.
    gripper_opening_measured: bool = True
    #: One feedback-controlled action for multiple arms on a shared timeline.
    supports_coordinated_control: bool = False

    @property
    def arm_names(self) -> Tuple[str, ...]:
        return tuple(arm.name for arm in self.arms)

    def arm(self, name: str) -> ArmInfo:
        for item in self.arms:
            if item.name == name:
                return item
        raise KeyError(name)


@dataclass(frozen=True)
class ArmState:
    """The state of one arm right now. Poses are in the frame the backend declares (BackendInfo.frame)."""

    pos: np.ndarray
    #: xyzw quaternion.
    quat: np.ndarray
    #: Gripper opening in meters; None without a reliable measurement.
    gripper_opening: Optional[float]
    #: The gripper's current target command: open / close.
    gripper_command: str


@dataclass(frozen=True)
class Snapshot:
    """Everything the backend can observe right now. Holds only what the Agent may see."""

    #: Budget used since reset (in BackendInfo.budget_unit).
    step: int
    arms: Dict[str, ArmState]
    #: Public camera name -> HxWx3 uint8. Remote frame striding may leave it empty in intermediate
    #: callbacks; the final results of reset, observe and actions must be complete.
    images: Dict[str, np.ndarray]


#: The stop_reason a backend reports when it used up the steps it was given. The MCP session
#: translates it into step_cap or budget, depending on which limit applied.
LIMIT = "limit"

COORDINATED_STOP_REASONS = (
    ("reached", "Every arm reached its target pose within tolerance; this does not mean the grippers are stable or holding."),
    ("stalled", "At least one arm kept failing to follow the shared target, so the whole joint action stopped."),
    ("command_applied", "Only gripper targets were executed; this does not mean the gripper is stable."),
    ("no_op", "Poses were already within tolerance and gripper targets did not change; the simulation did not advance."),
    ("ended", "The environment stopped accepting actions."),
)


@dataclass
class MotionReport:
    """The result of one move_to / set_gripper."""

    #: The backend's own stop reason (declared in BackendInfo), or LIMIT.
    stop_reason: str
    executed_steps: int
    #: The observation when the action ended.
    snapshot: Snapshot
    #: Move only: meters still missing to the target position.
    remaining_distance_m: float = 0.0
    #: Calibration records for the journal only (per-step displacement, residuals), never returned to the Agent.
    telemetry: Dict[str, Any] = field(default_factory=dict)
    #: Whitelisted per-arm feedback for coordinated actions; no privileged signals.
    arm_results: Dict[str, Any] = field(default_factory=dict)


#: Called once per step; the MCP session uses it to record frames live and poll success.
StepCallback = Callable[[Snapshot], None]


class Backend:
    """The interface a backend implements. Only signatures and rules here; see `stepped.py`, `fake.py` and `libero.py` for implementations.

    Rules:
    - All poses are in the frame declared by `info.frame`; quaternions are xyzw.
    - `move_to` / `set_gripper` run at most `max_steps` steps (in budget units) and call
      `on_step` once per step; if they use them all without stopping they report `LIMIT`, and
      the session decides whether to tell the Agent step_cap or budget.
    - Budget checks (how much is left, whether to refuse) happen in the session; backends
      do not deal with them.
    - When `on_step` is called, `success()` must already reflect the state after that step: the
      session polls success in the callback and records the first step at which the task was
      solved. Calling back first and updating success afterwards would miss that moment.
    - `success()` is for evaluation only; the session guarantees it never reaches a tool result.
    """

    @property
    def info(self) -> BackendInfo:
        raise NotImplementedError

    @property
    def catalogue(self) -> "Catalogue":
        raise NotImplementedError

    def reset(self, task: TaskSpec, variant: int) -> Snapshot:
        raise NotImplementedError

    def observe(self) -> Snapshot:
        raise NotImplementedError

    def move_to(
        self,
        arm: str,
        pos: np.ndarray,
        quat: np.ndarray,
        *,
        max_steps: int,
        on_step: Optional[StepCallback] = None,
    ) -> MotionReport:
        raise NotImplementedError

    def set_gripper(
        self,
        arm: str,
        close: bool,
        *,
        max_steps: int,
        on_step: Optional[StepCallback] = None,
    ) -> MotionReport:
        raise NotImplementedError

    def control_arms(self, targets, *, max_steps: int, on_step=None) -> MotionReport:
        """Absolute pos/xyzw quat and optional gripper goals on a shared timeline.

        Validate everything before acting. Omitted arms hold their initial pose.
        Each joint environment action costs one step regardless of arm count.
        """
        raise McpToolError("unsupported_operation", "This backend does not support coordinated control of several arms")

    def success(self) -> bool:
        raise NotImplementedError

    def end_episode(self) -> None:
        """End the current episode and release the simulator. The next reset builds a new one."""
        raise NotImplementedError

    def close(self) -> None:
        """Release the whole backend (including remote subprocesses and connections), not just one episode."""
        self.end_episode()

    def drain_transport_events(self) -> List[Dict[str, Any]]:
        """Per-call transport statistics for the local journal only; never returned to the Agent."""
        return []

    def journal_params(self) -> Dict[str, Any]:
        """The control constants of this run, written to the journal's controller_params event for later checks."""
        return {}

    def episode_audit(self) -> Optional[Dict[str, Any]]:
        """A physics check-up at the end of the episode (such as maximum penetration depth), for the journal summary only, never the Agent.

        Most backends have none and return None.
        """
        return None

    def extra_tools(self) -> List[Dict[str, Any]]:
        """Tools the backend adds itself (in MCP tool-list format). Most backends have none.

        The session only forwards and logs them; argument validation, descriptions and the
        information boundary of these tools are the backend's responsibility, and their results
        must not contain `info.forbidden_terms` either.
        """
        return []

    def call_extra(self, name: str, args: Dict[str, Any], *, running: bool) -> "ExtraResult":
        """Run a tool declared in `extra_tools`. `running` says whether an episode is running right now."""
        raise McpToolError("unknown_tool", f"Unknown tool: {name}")


@dataclass
class ExtraResult:
    """The result of a backend tool: data and images for the Agent, plus fields that go only to the log."""

    data: Dict[str, Any]
    images: List[np.ndarray] = field(default_factory=list)
    #: Extra fields for the session log (`<output_root>/extra_tools.jsonl`).
    log: Dict[str, Any] = field(default_factory=dict)


class Catalogue:
    """Task catalogue: which tasks the Agent can choose, how it chooses them, and how the chosen one is described.

    It is separate from the backend because which tasks exist is a property of the benchmark,
    unrelated to how the arm moves; LIBERO and FakeSim share one implementation
    (`catalogue.SuiteCatalogue`) with different descriptions.
    """

    def texts(self) -> "CatalogueTexts":
        raise NotImplementedError

    def resolve(self, args: Dict[str, Any], *, locked: bool) -> Tuple[TaskSpec, int]:
        """Pick the task and variant from reset_task arguments; raise McpToolError if they are invalid."""
        raise NotImplementedError

    def commit(self, task: TaskSpec, variant: int) -> None:
        """Remember the task actually used, so a later reset_task without arguments keeps it."""
        raise NotImplementedError

    def task_payload(self) -> Dict[str, Any]:
        """The current task as the Agent sees it (the task field of reset_task / get_session_info)."""
        raise NotImplementedError

    def idle_payload(self) -> Dict[str, Any]:
        """The default-task fields of get_session_info before the first reset."""
        raise NotImplementedError

    def listing(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """The result of list_tasks."""
        raise NotImplementedError


@dataclass(frozen=True)
class CatalogueTexts:
    """Catalogue-related pieces of the tool descriptions."""

    #: Opening sentence of the reset_task description on omitting arguments (ends with a newline).
    reset_intro: str
    #: Paragraph of the reset_task description introducing what can be chosen (ends with a newline, or empty).
    reset_blurb: str
    #: reset_task parameters (JSON Schema properties).
    reset_properties: Dict[str, Any]
    #: Paragraph of the reset_task description on switching tasks.
    reset_howto: str
    #: The fields of the task value in results, such as "suite, task_index, name, instruction, init_state_index".
    task_fields: str
    #: Full description of list_tasks.
    list_description: str
    #: list_tasks parameters.
    list_properties: Dict[str, Any]
    #: Name of the default-task field get_session_info returns while idle.
    default_field: str = "default_scene"


def number_word(n: int) -> str:
    """Counts in tool descriptions ("two PNGs", "four reasons"): words for 1..10, digits beyond."""
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    return words.get(n, str(n))


def join_names(names: Sequence[str], conjunction: str = "and") -> str:
    """["a", "b", "c"] -> "a, b and c"."""
    items: List[str] = list(names)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {conjunction} " + items[-1]
