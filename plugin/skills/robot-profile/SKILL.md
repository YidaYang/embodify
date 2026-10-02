---
name: robot-profile
description: Create, update and read a written profile of the robot body you control — arms, grippers, cameras, frames, calibration and measured motion behavior. Use when starting work with a new robot or backend, and whenever you learn something lasting about the body.
---

# Robot profile

A profile is a short Markdown file holding what you know about one robot body,
so that you and later sessions do not have to rediscover it.

## Where it lives

`.embodify/robots/<robot-id>.md` in the current workspace. `<robot-id>` names
the backend and the body, for example `libero-panda` or `robodojo-arx-x5`.
Create the directory if needed. If the user keeps profiles somewhere else, use theirs.

## Creating a profile

1. Copy [template.md](template.md).
2. Fill the static facts from `get_session_info`, the tool schemas and the tool
   descriptions: arms, controllable axes, gripper, cameras, units, frames, budget.
3. Fill measured facts only from observations you made, with the date: how far
   a 5 cm move actually went, how many steps a gripper close took, which camera
   shows the table edge best. Mark anything you have not checked as `unverified`.
4. Record calibration (camera intrinsics and extrinsics, TCP offset) only when a
   tool returned it or the user supplied it. Never invent numbers.

## Using a profile

Read it before the first action. When an observation contradicts the profile,
trust the observation, then correct the profile and date the change.

Keep it to facts that change how you act. Stories about individual episodes
belong in the `task-experience` skill.
