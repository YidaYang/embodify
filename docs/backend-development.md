# Writing a backend

A backend connects Embodify to one simulator, benchmark or robot. The MCP server
handles sessions, budgets, logging and the tools the agent sees; the backend
handles the robot. Start from `embodify_mcp/backend/base.py` and the reference
implementations in `embodify_mcp/backend/fake.py`.

## Describe the robot

`BackendInfo` declares arms, cameras, coordinate frame, step budget and
supported operations. The tool descriptions the agent reads are generated from
it, so be precise, and never silently ignore an axis, arm or gripper the
backend cannot control. A `Catalogue` resolves task identity; the shared
`TaskSpec` type lives in `embodify_mcp/types.py`.

## Implement the interface

Implement `info`, `catalogue`, `reset`, `observe`, `move_to`, `set_gripper`,
`end_episode` and `close`. Implement `success` only when `has_success` is true.

- `observe` and every action return a `Snapshot` with the actual arm state and
  camera images.
- Actions return a `MotionReport` with the steps actually executed and a stop reason.
- Run the low-level control loop inside the backend, next to the simulator or
  robot, so each tool call is a single round trip.
- Call `on_step` after updating success for that step; the final callback must
  describe the final state.
- Respect the execution budget, and release resources on EOF, errors and normal stop.
- Never retry a non-idempotent motion automatically.
- Import simulator and GPU dependencies lazily, so importing the core never loads them.

Evaluation results go to the journal, never to the agent. Extra tools are
optional: validate their inputs explicitly and keep evaluator fields out of
their results. Enable `control_arms` once the full validation and feedback
contract in the [action contract](action-contract.md) is implemented.

## Register and test

Add the backend to the factory and backend list in `embodify_mcp/mcp.py` and to
the parametrized contract suite in `tests/test_backend_contract.py`. Test
invalid inputs (which must not move anything), partial execution, repeated
reset and stop, disconnects and cleanup. Then document its setup in
[backend setup](backends.md).

Hardware backends also need tested watchdogs, an independent emergency stop and
exclusive controller ownership; see [security](../SECURITY.md).
