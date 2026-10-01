# Backend setup

The core package runs in any Python 3.8+ environment and needs only NumPy.
Each simulator lives in its own environment, installed following its upstream
instructions. Install Embodify into that environment as well, from source or
from a wheel built with `python -m build`. For a remote backend, install it on
the remote machine too.

## Server options

| Option | Backends | Meaning |
|---|---|---|
| `--backend` | all | `fake`, `fake-two-arm`, `libero`, `robodojo` or `remote` |
| `--output-root` | all | Where run logs and images are written (default `out/mcp`) |
| `--monitor-port` | all | Also serve the live monitor and replay page on this localhost port |
| `--max-steps-per-call` | all | Most internal steps one tool call may run (default 30) |
| `--lock-task` | all | Fix the scene, so the agent cannot choose another task or initial state |
| `--expected-episode-budget` | all | Refuse to start if the backend's episode budget differs |
| `--image-size`, `--max-steps` | fake, libero | Camera resolution; step budget per episode |
| `--suite`, `--task-index`, `--init-state` | fake, libero | Default scene when the agent does not choose one |
| `--controller-verified` | fake, libero | Record that you checked the controller calibration |
| `--robodojo-config` | robodojo | RoboDojo configuration file |
| `--remote-config`, `--frame-stride` | remote | Transport configuration; send every n-th intermediate image (every robot state is kept) |

## Fake

`fake` (one arm, two cameras) and `fake-two-arm` (two arms, three cameras) are
kinematic simulations that need nothing beyond the core package. Use them to
try the tools, set up your agent host and run CI.

## LIBERO

LIBERO runs on Linux or WSL2. Install
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) with its task files,
initialization states and assets; demonstration datasets are not needed.

Reference environment: Python 3.8, NumPy 1.24.4, robosuite 1.4.1, MuJoCo 3.2.3,
CPU PyTorch 2.1.2 (to load initialization states) and the `libero` 0.1.1
distribution. The complete dependency set is in
[environments/libero-py38-constraints.txt](../environments/libero-py38-constraints.txt);
it describes the LIBERO environment, not the agent host.

```sh
# Inside the LIBERO environment
python -m pip install /path/to/embodify
MUJOCO_GL=egl embodify-mcp --backend libero --output-root out/libero
```

Use `MUJOCO_GL=egl` on a GPU machine or `MUJOCO_GL=osmesa` to render on the CPU.
The agent can browse all 130 tasks of the five suites with `list_tasks` and
start any of them with `reset_task`; add `--lock-task` to fix one scene for
controlled experiments.

Before setting `--controller-verified`, check translation and rotation scale,
gripper sign and end-effector tracking on your installation. Report the LIBERO
source revision you installed together with your results. To run the contract
tests against the real simulator:

```sh
EMBODIFY_LIBERO_CONTRACT=1 python -m pytest tests/test_backend_contract.py -k libero
```

## RoboDojo

The RoboDojo backend offers all 54 RoboDojo simulation tasks: 42 base tasks and
their 12 `*_random` variants. Every task uses the dual ARX X5 arms, the three RGB
cameras `cam_head`, `cam_left_wrist` and `cam_right_wrist`, and one environment.
In `imitate_sorting_sequence`, `make_kong` and `play_tic_tac_toe`, RoboDojo also
moves a scripted Franka arm that the agent does not control.

The agent browses the tasks with `list_tasks`, which gives each task's name,
instruction and episode budget, and picks one with `reset_task`'s `task`
argument. Four tasks (`classify_objects_by_language`, `general_pickup`,
`pour_by_language` and `stack_blocks_by_language`) generate their instruction
from each episode's layout; `list_tasks` shows it as `null` and `reset_task`
returns the actual instruction. Run the server with `--lock-task` to keep the
agent on the configured task.

**Source.** Clone [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) at
`726e9aabfaa642203722eb126f5eaf0f37f3e1ad` and initialize its submodules at the
revisions pinned in `embodify_mcp/robodojo/config.py`: XPolicyLab
`bb9a0b5f5136a74503b679af830bfd0a3a837d5c`, IsaacLab
`afca7b09d60d8beb9c1cb28b43066499940b969b` and cuRobo
`d17b54ce32cba095c0b000c4c58777075d11de0e`. Install the runtime and assets
following the upstream instructions and terms.

**Runtime.** Reference environment: Python 3.11, NumPy 1.26.0,
PyTorch 2.7.0+cu128, Isaac Sim 5.1.0.0, Isaac Lab 0.54.3 (headless build, see
below) and warp-lang 1.11.0. Isaac Sim needs an NVIDIA RTX GPU; GPUs without RT
cores, such as A100 and H100, are not supported upstream.

**Headless Isaac Lab.** Isaac Lab's package pins `starlette==0.49.1`, which
conflicts with the Isaac Sim stack and is not needed headless. Build a copy
without that constraint and install it:

```sh
python tools/prepare_isaaclab_compat.py \
  --source /path/to/IsaacLab/source/isaaclab \
  --output /path/to/isaaclab-headless
```

Keep the output outside the pinned checkout. The helper copies the runtime
Python files and records their hashes; it does not install or redistribute
Isaac Lab. Do not use this build for livestreaming.

**Configure and run.** Copy [examples/robodojo.json](../examples/robodojo.json)
to `robodojo.local.json` and set absolute paths for `source`, `python` and
`output`. `task` is the task a plain `reset_task` starts (default
`stack_blocks`). The other fields are `seed` (seed group 0, 1 or 2),
`layout_index`, `image_size`, `startup_timeout_s`, `action_timeout_s` and
`episode_budget`: environment actions per episode. With `null`, each task keeps
its native RoboDojo limit, from 200 (`align_blocks`, `general_pickup`) to 1900
(`fasten_screws`); an integer overrides it for every task. Raise it for longer
agent sessions and report the value with your results. Each run's summary
records the budget used, the task's native limit and whether it was overridden.
`--expected-episode-budget` compares against the configured task's budget.

```sh
python -m pip install ".[diagnostics]"
embodify-mcp-preflight --source /path/to/RoboDojo --runtime
embodify-mcp --backend robodojo --robodojo-config /path/to/robodojo.local.json
```

Run the preflight in the Isaac worker environment: it checks the source
revisions and runtime packages, and `--assets` also verifies the assets against
a `.embodify-assets-manifest.json` you generate from the pinned asset source.
It also checks every task's files and compares each task's step limit,
instruction and robot setup with the catalogue the server advertises; pass
`--task NAME` (repeatable) to check only some tasks. A source archive's
`.embodify-source-manifest.json` must list the files of each task you run. Each
`reset_task` starts a worker process for the chosen task, which can take a few minutes;
`stop_episode` or a lost connection closes it, and on Linux the worker exits
together with its supervisor.

## Remote backends

Run the simulator on another machine, typically a GPU server, while the MCP
server stays next to your agent. Run logs and the monitor page stay on your
machine. The MCP server starts `embodify-mcp-backend` on the remote side,
usually over SSH, and talks to it over stdio; `embodify-mcp-backend` accepts the
same backend options as `embodify-mcp`.

Try it locally with a Fake backend in a subprocess:

```sh
embodify-mcp --backend remote --remote-config examples/remote-stdio.json
```

For a GPU server, start from [examples/remote-ssh.json](../examples/remote-ssh.json):
`robot-sim` is an alias from your `~/.ssh/config`, and the `/opt/...` paths stand
for your installation. Install Embodify and the simulator runtime on the server,
set up key-based SSH and verify the host key before starting your agent. Keep
passwords and private keys out of the JSON.

| Field | Meaning |
|---|---|
| `transport` | `stdio` (with `command`, optionally `cwd`) or `tcp` (with `host` and `port`) |
| `timeout_s` | Deadline for one call |
| `connect_timeout_s` | Deadline for the startup handshake |
| `heartbeat_timeout_s` | Heartbeats keep slow links alive; a silent link aborts the episode |
| `close_timeout_s` | Grace period when shutting down |
| `frame_stride` | Send every n-th intermediate image |

If the connection drops, the episode aborts and the next `reset_task`
reconnects. RoboDojo workers can take minutes to start, so set your agent
host's MCP tool timeout to cover it; the example allows 900 seconds per call.
Heartbeats are not MCP progress notifications, so the host timeout still applies.

`embodify-mcp-backend --listen-port PORT` serves over TCP instead of stdio. The
TCP transport has no authentication or encryption: use it only on localhost or
through an SSH tunnel.
