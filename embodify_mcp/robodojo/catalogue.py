"""RoboDojo task catalogue: the Agent picks a task by name; layout and seed group stay with the operator."""
from ..backend.base import Catalogue, CatalogueTexts, McpToolError
from ..sim.base import TaskSpec
from .config import DEFAULT_TASK
from .tasks import TASK_NAMES, TASKS, episode_budget

GENERATED_NOTE = 'The instruction is generated from the object layout of each episode; use the one reset_task returns'
SUPPORT_ARM_NOTE = 'The scene has an extra Franka arm scripted by the environment; the Agent controls only left and right'


class TaskCatalogue(Catalogue):
    """`offered` narrows the startable tasks (one per worker); TaskSpecs are identical either way."""

    def __init__(self, default_task=DEFAULT_TASK, *, episode_budget_override=None, offered=None):
        names = TASK_NAMES if offered is None else tuple(offered)
        if default_task not in names or not set(names) <= set(TASKS):
            raise ValueError('unknown RoboDojo task')
        self._tasks = {name: TaskSpec(suite='robodojo', task_index=TASK_NAMES.index(name), name=name,
                                      instruction=TASKS[name].instruction or '', n_init_states=1,
                                      episode_budget=episode_budget(name, episode_budget_override))
                       for name in names}
        self._current = self._tasks[default_task]
        self._instruction = None  # what the running episode's environment reported

    @property
    def task(self):
        return self._current

    def texts(self):
        return CatalogueTexts(
            reset_intro='**The argument is optional**: omit task to keep the current task (initially the server default), which is the most common use.\n',
            reset_blurb='Every reset starts a new simulator process; the first load can take several minutes.\n',
            reset_howto=('To switch tasks, use list_tasks first to see the name, instruction and max_steps of each task, '
                         'then pass the chosen name as task. The operator configures the object layout; the Agent cannot change it. '
                         'If the server was started with --lock-task (task_locked is true in get_session_info), passing task '
                         'returns a task_locked error.\n'),
            reset_properties={'task': {
                'type': 'string',
                'description': 'RoboDojo task name (a name from list_tasks); omit to keep the current task, initially the server default.'}},
            task_fields='name, instruction',
            list_description=('List the RoboDojo tasks this server offers. Takes no arguments and starts no simulation. Each entry gives '
                              'name, instruction and max_steps (the environment action budget per episode of that task). For tasks with '
                              'a null instruction, the instruction is generated from the object layout of each episode; use the instruction '
                              'reset_task returns. Some tasks carry a note that explains such generated instructions, or an extra arm '
                              'controlled by the environment.'),
            list_properties={},
            default_field='default_task')

    def resolve(self, args, *, locked):
        if 'task' in args and locked:
            raise McpToolError('task_locked', "This run's task is fixed (the server was started with --lock-task), so task is not "
                                              'accepted. Call reset_task without arguments.')
        if set(args) - {'task'}:
            raise McpToolError('invalid_input', 'reset_task accepts only task; the operator configures the object layout')
        if 'task' not in args:
            return self._current, 0
        name = args['task']
        if not isinstance(name, str):
            raise McpToolError('invalid_input', 'task must be a string')
        if name not in self._tasks:
            raise McpToolError('invalid_input', 'No task named %r; see list_tasks for the available tasks.' % name)
        return self._tasks[name], 0

    def commit(self, task, variant):
        if variant != 0 or self._tasks.get(task.name) != task:
            raise McpToolError('invalid_input', 'The task does not belong to this server')
        self._current, self._instruction = task, None

    def accept_instruction(self, instruction):
        """Record the running episode's instruction; a static one must equal the pinned template."""
        if not isinstance(instruction, str) or not instruction.strip():
            raise RuntimeError('Environment reported no instruction')
        static = TASKS[self._current.name].instruction
        if static is not None and instruction != static:
            raise RuntimeError('Environment instruction differs from the pinned task instruction')
        self._instruction = instruction

    def task_payload(self):
        return {'name': self._current.name,
                'instruction': self._instruction or TASKS[self._current.name].instruction}

    def idle_payload(self):
        return {'default_task': self.task_payload()}

    def listing(self, args):
        if args:
            raise McpToolError('invalid_input', 'list_tasks takes no arguments')
        rows = []
        for task in self._tasks.values():
            row = {'name': task.name, 'instruction': TASKS[task.name].instruction, 'max_steps': task.episode_budget}
            if row['instruction'] is None:
                row['note'] = GENERATED_NOTE
            elif TASKS[task.name].support_arm:
                row['note'] = SUPPORT_ARM_NOTE
            rows.append(row)
        return {'ok': True, 'tasks': rows}
