# Security

## Connecting hardware

Embodify's current backends drive simulators. Before connecting a physical
robot, add safety limits that do not depend on the agent: workspace bounds,
speed and force limits, a watchdog, exclusive control of the robot and an
emergency stop.

Tool calls run one at a time. `stop_episode` ends an episode but does not
interrupt a call that is already executing. A remote movement that times out
may have partly executed: the session aborts, and the movement must not be
repeated automatically. Observe again or reset.

## Network exposure

The backend TCP transport and the monitor page have no authentication or
encryption, and listen on localhost by default. Reach remote machines over
stdio and SSH or through a trusted network you control, and never expose these
ports to the Internet. Verify SSH host keys. A remote configuration runs the
command it lists, so keep configuration files out of reach of untrusted agents.

## Fair evaluation

Tool results never include task-success signals; run logs and the monitor
page do. An agent's own shell and file tools can read files in its
workspace, so run evaluations under a separate user, container or machine that
cannot read evaluator logs, reference solutions or simulator ground truth.
Start the server with `--lock-task` so the agent cannot choose another task or
initial state, and keep the results of earlier runs out of held-out evaluations.

## Logs

Logs contain camera images and diagnostics. Review them before sharing.

## Reporting a vulnerability

Please report vulnerabilities privately with GitHub's "Report a vulnerability"
button on the repository's Security tab, not in a public issue. Leave secrets
and personal data out of the report.
