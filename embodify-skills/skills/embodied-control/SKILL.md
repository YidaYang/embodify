---
name: embodied-control
description: Operate a robot through the Embodify MCP tools (get_session_info, reset_task, observe, move_relative, set_gripper, control_arms, stop_episode). Use whenever you are asked to perform, attempt or debug a manipulation task with these tools.
---

# Embodied control loop

You control a robot through MCP tools. Each action call runs one bounded motion
next to the simulator or robot and returns what actually happened, with fresh
camera images. Treat every result as a measurement, not a confirmation.

## Before the first action

1. Call `get_session_info` and read the tool descriptions: arm names, camera
   names, units, frames and step budget. Do not assume them from another robot.
2. If a robot profile exists, read it (skill `robot-profile`). If lessons exist
   for this robot or task, read them (skill `task-experience`).
3. Call `reset_task`, look at every camera image, then write a short plan: where
   the target is, how you will approach it and how you will check each stage.

## Each action

- Near objects, move a few centimeters at a time. Use larger moves only in free space.
- After every call, read the stop reason, the actual displacement and the remaining error:
  - `reached`: the pose is within tolerance. It says nothing about grasping or placing.
  - `stalled`: progress stopped (contact, limit or planning failure). Observe and
    replan; do not repeat the same command.
  - `step_cap` / `budget`: execution was cut short. Plan from where the arm is now.
  - `command_applied`: the gripper command was sent; it may not have settled.
  - `no_op`: nothing needed to move. `ended`: the episode no longer accepts actions.
- Never resend the original full delta after partial progress. Compute a new
  delta from the current state.
- Judge depth from more than one camera. Use the wrist camera for final alignment.

## Grasping and releasing

- Open the gripper before approaching. Approach above the object, descend, then
  close the gripper in a separate call.
- Check a grasp by lifting a little and looking at the images (and at the
  gripper opening, if the backend reports one). A closed gripper is not proof of a grasp.
- In `control_arms`, a gripper target applies from the first step of the motion,
  not at its end. To place an object, move to the place pose in one call, then
  open the gripper in the next call.

## Ending

- Call `stop_episode` when the task is done or further attempts are pointless.
  Report what you observed, not what you intended.
- Do not read log, journal or evaluator files to learn whether you succeeded.
  Success signals are withheld from you on purpose; judge from observations.
- Afterwards, record what you learned with the `task-experience` skill.
