---
name: task-experience
description: Keep a written record of manipulation experience — read relevant lessons before a robot task, write one short lesson after each episode, and turn repeated lessons into rules. Use before starting and after finishing any robot task.
---

# Task experience

The simplest form of self-improvement: learn from every episode in writing, and
apply it next time.

## Before a task

Read `.embodify/experience/rules.md` if it exists, and the lesson files whose
robot or task matches the current one. State which lessons you will apply.

## After each episode

Append one entry to `.embodify/experience/<robot-id>/<task-slug>.md`:

~~~markdown
## <YYYY-MM-DD> <run id> — success | failure | unknown
- Goal:
- What happened: key steps and the observations that mattered
- What went wrong or right:
- Lesson: one actionable sentence, e.g. "open the gripper in a separate call after reaching the place pose"
- Next time:
~~~

Record the outcome as you observed it (images, robot state) or as the user told
you. Do not take it from log or evaluator files.

## Turning lessons into rules

When the same lesson appears in two or more episodes, add it to
`.embodify/experience/rules.md` as one short rule, with links to the entries
that support it. Delete a rule when later evidence contradicts it.

If your host has its own memory, also save a one-line pointer there so that
future sessions know this experience directory exists.

## When results are compared

For benchmarks or papers, note whether experience files were read during a run:
they change what the agent knows before it starts.
