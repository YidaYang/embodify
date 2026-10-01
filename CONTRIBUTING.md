# Contributing to Embodify

Contributions are welcome: new simulators and robots, new skills, fixes and
documentation. By taking part you agree to follow the
[code of conduct](CODE_OF_CONDUCT.md).

## Set up

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python tools/check_release.py
```

Core tests run without a GPU, simulator downloads, credentials or network
access. To check an installed package, run `embodify-mcp-smoke` from outside
the checkout.

## Adding a backend

To request support for a simulator or robot, open an issue with the backend
request form. To add one yourself, follow [writing a backend](docs/backend-development.md).
Add the backend to the contract test suite, document its setup in
[backend setup](docs/backends.md) and add it to the backend table in both
READMEs. Test changes to a simulator or robot backend on that backend, not only
on Fake.

## Adding a skill

Create `embodify-skills/skills/<skill-name>/SKILL.md` with `name` and
`description` in its frontmatter, and keep supporting files such as templates
in the same folder. Write for the agent: concrete steps, what to check after
each action and what to avoid. If you use Claude Code, check the skills pack
with `claude plugin validate --strict embodify-skills/.claude-plugin/plugin.json`.
List the new skill in both READMEs.

## Changing action semantics

Changes to motion, grippers or stop reasons need contract tests. Validate the
whole input before moving any arm, never replay a timed-out movement
automatically, and document units, frames and stop reasons in the
[action contract](docs/action-contract.md).

## Pull requests

- Keep simulator and GPU imports lazy, so importing the core never loads them.
- Put backend-specific dependencies in that backend's setup instructions, not in
  the core package dependencies.
- Do not commit credentials, personal host configuration, run logs or simulator assets.
- Preserve third-party notices.

Contributions are licensed under Apache-2.0 unless a file states otherwise.
