# Action and protocol contract

## Protocol

Transport: newline-delimited JSON-RPC over stdio. Supported version identifiers:
2024-11-05, 2025-03-26, 2025-06-18, 2025-11-25. An unknown client version selects
2025-11-25; the client must disconnect if it cannot use that version. This is a
synchronous tools-only implementation; optional tasks, subscriptions, sampling,
HTTP MCP transport and cancellation are not implemented or advertised.
The backend TCP protocol is a separate internal versioned protocol, not MCP HTTP.

Malformed JSON uses -32700; invalid requests -32600; invalid params -32602;
unknown methods -32601. Tool execution failures use `isError`. Operator stderr
may contain diagnostics; public unexpected-error text is sanitized. Input size
is bounded to 1 Mi characters per line. Notifications cannot dispatch actions.
Text and structured results are provided, plus inline PNG image content.

## State and motion

One server process owns one backend and at most one active episode. Actions are
serialized. Reset opens a new episode, stop closes it; reset after stop is valid.
Failed remote connections invalidate the old episode; no transparent resume.

Use `get_session_info` and the returned tool schemas, not assumptions about
arm or camera names. Single-arm state is flat; multi-arm state uses `arms`.
`move_relative` translation is meters in the backend-declared fixed frame.
`delta_rpy` is radians about fixed XYZ axes: Rz(yaw) Ry(pitch) Rx(roll), left
multiplied onto the starting orientation. Quaternions are normalized xyzw.
LIBERO uses its robot base frame; the RoboDojo pilot uses the scene/world axes
and requires a single environment at origin zero.

A call executes up to its action budget; inspect actual displacement, remaining
error and stop reason before planning the next action. Never resend the full
original delta after partial progress. `observe` does not advance these
simulation backends while the agent thinks; that is not a promise for a live robot.

For coordinated motion, all arms are validated before dispatch. Unspecified
arms hold their start pose, omitted gripper targets retain the prior command.
Translation is linear and orientation follows shortest-arc SLERP with common
progress. Any lagging arm may stall the whole action. A gripper target applies
from the first step, not after motion. This is pose coordination, not force
coordination or a guarantee of simultaneous physical contact.

## Feedback

- `reached`: pose tolerance reached, no grasp/place guarantee.
- `stalled`: progress insufficient; re-observe and replan.
- `command_applied`: command sent, not necessarily settled.
- `step_cap` / `budget`: bounded execution stopped; inspect partial result.
- `no_op`: no state-changing motion required.
- `ended`: environment stopped accepting actions.

RoboDojo gripper opening is unknown (`null`); its normalized upstream target is
not a measurement. Its budget counts `take_action` calls, not MCP calls or
physics ticks. By default each episode gets its task's native RoboDojo limit
(550 for `stack_blocks`, 200 to 1900 across tasks) and each tool call at most 30;
overrides are operator configuration and must be recorded when comparing benchmarks.

Task-success signals are evaluator-only. Journal `success`, first-success step
and success-at-stop describe different observations; an earlier success can
become false before stop. Hardware integrations must define independent limits
and stop behavior; neither the host nor these tool descriptions enforce safety.
