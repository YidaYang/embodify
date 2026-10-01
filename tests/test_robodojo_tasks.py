"""Task selection over the RoboDojo catalogue, with CPU doubles; no Isaac/PhysX validation."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodify_mcp.backend.base import McpToolError
from embodify_mcp.backend.robodojo import RoboDojoBackend
from embodify_mcp.backend.wire import pack_task, unpack_task
from embodify_mcp.mcp import McpSession
from embodify_mcp.robodojo import preflight
from embodify_mcp.robodojo.catalogue import TaskCatalogue
from embodify_mcp.robodojo.config import REVISION, SUBMODULES, Settings
from embodify_mcp.robodojo.episode import EpisodeBackend, info
from embodify_mcp.robodojo.isaac_driver import IsaacDriver
from embodify_mcp.robodojo.source import verify_source
from embodify_mcp.robodojo.tasks import COMMON_FILES, CONVEYOR_TASKS, TASK_NAMES, TASKS, task_files
from embodify_mcp.types import TaskSpec
from tests.test_remote_backend import service
from tests.test_robodojo_backend import GENERATED, Driver

TESTS = Path(__file__).resolve().parent
LANGUAGE = ('classify_objects_by_language', 'general_pickup', 'pour_by_language', 'stack_blocks_by_language')
SUPPORT_ARM = ('imitate_sorting_sequence', 'make_kong', 'play_tic_tac_toe')


def worker_command(**driver):
    # The supervisor appends `--task NAME`; the double runs the task it was started for.
    code = ('import sys;sys.path.insert(0,' + repr(str(TESTS)) + ');'
            'from test_robodojo_backend import Driver;'
            'from embodify_mcp.robodojo.episode import EpisodeBackend;'
            'from embodify_mcp.backend.serve import serve_connection;'
            'from embodify_mcp.backend.transport import LineTransport;'
            'task=sys.argv[sys.argv.index("--task")+1];'
            'serve_connection(EpisodeBackend(Driver(task=task,**' + repr(driver) + ')),'
            'LineTransport(sys.stdin.buffer,sys.stdout.buffer,lambda:None))')
    return [sys.executable, '-u', '-c', code]


def supervisor(tmp_path, *, task=None, episode_budget=None, **driver):
    config = {'source': 'unused', 'python': sys.executable, 'output': str(tmp_path)}
    if task is not None: config['task'] = task
    if episode_budget is not None:
        config['episode_budget'] = episode_budget
        driver.setdefault('budget', episode_budget)
    path = tmp_path/'robodojo.json'
    path.write_text(json.dumps(config))
    return RoboDojoBackend(path, worker_command=worker_command(**driver))


def call(session, tool, args=None):
    result = session.call_tool(tool, args or {})
    assert not result.get('isError'), result
    return result['structuredContent']


def test_upstream_facts_in_the_table():
    # 42 base tasks and 12 `_random` variants; each variant is its own module and config.
    assert len(TASK_NAMES) == 54 == len(set(TASK_NAMES))
    assert len([n for n in TASK_NAMES if n.endswith('_random')]) == 12
    assert tuple(n for n in TASK_NAMES if TASKS[n].instruction is None) == LANGUAGE
    assert tuple(n for n in TASK_NAMES if TASKS[n].support_arm) == SUPPORT_ARM
    assert TASKS['stack_blocks'].step_limit == 550 and TASKS['fasten_screws'].step_limit == 1900


def test_listing_names_every_task_with_public_instruction_and_budget():
    listing = TaskCatalogue().listing({})
    rows = {row['name']: row for row in listing['tasks']}
    assert list(rows) == list(TASK_NAMES)
    for name, row in rows.items():
        assert row['max_steps'] == TASKS[name].step_limit
        assert row['instruction'] == TASKS[name].instruction
        assert ('note' in row) is (name in LANGUAGE + SUPPORT_ARM)
    assert rows['stack_blocks']['instruction'] == 'Stack the three blocks with different textures.'
    assert rows['stack_blocks_by_language']['instruction'] is None
    text = json.dumps(listing, ensure_ascii=False).lower()
    assert all(term not in text for term in info(550).forbidden_terms + ('success', 'privileged'))
    overridden = TaskCatalogue(episode_budget_override=10000).listing({})['tasks']
    assert {row['max_steps'] for row in overridden} == {10000}
    with pytest.raises(McpToolError):
        TaskCatalogue().listing({'suite': 'robodojo'})


def test_resolve_selects_rejects_and_locks():
    catalogue = TaskCatalogue('push_T')
    task, variant = catalogue.resolve({}, locked=False)
    assert (task.name, variant, task.episode_budget) == ('push_T', 0, 600)
    task, _ = catalogue.resolve({'task': 'fasten_screws'}, locked=False)
    assert task.name == 'fasten_screws' and task.episode_budget == 1900
    catalogue.commit(task, 0)
    assert catalogue.resolve({}, locked=False)[0].name == 'fasten_screws'  # a plain reset keeps the choice
    for args in ({'task': 'no_such_task'}, {'task': 3}, {'task': 'push_T', 'seed': 1}, {'layout_index': 2}):
        with pytest.raises(McpToolError) as caught:
            catalogue.resolve(args, locked=False)
        assert caught.value.code == 'invalid_input'
    with pytest.raises(McpToolError) as caught:
        TaskCatalogue('push_T').resolve({'task': 'push_T'}, locked=True)
    assert caught.value.code == 'task_locked'
    assert TaskCatalogue('push_T').resolve({}, locked=True)[0].name == 'push_T'
    with pytest.raises(McpToolError):  # a worker only starts its own task
        TaskCatalogue('push_T', offered=('push_T',)).commit(TaskCatalogue().resolve({}, locked=False)[0], 0)


def test_operator_settings_choose_default_task():
    with pytest.raises(ValueError, match='task'):
        Settings(source='s', python='p', output='o', task='stack_everything')
    example = Settings.load(TESTS.parent/'examples/robodojo.json')
    assert example.task == 'stack_blocks' and example.episode_budget is None


def test_static_instruction_must_match_pinned_template():
    backend = EpisodeBackend(Driver(task='push_T', instruction='Stack the three blocks with different textures.'))
    with pytest.raises(RuntimeError, match='instruction'):
        backend.reset(backend.catalogue.task, 0)


@pytest.mark.parametrize('instruction', ['', '   '])
def test_generated_instruction_must_exist(instruction):
    backend = EpisodeBackend(Driver(task='pour_by_language', instruction=instruction))
    with pytest.raises(RuntimeError, match='instruction'):
        backend.reset(backend.catalogue.task, 0)


def test_generated_instruction_comes_from_the_environment(tmp_path):
    backend = EpisodeBackend(Driver(task='stack_blocks_by_language'))
    session = McpSession(backend, output_root=tmp_path)
    try:
        assert call(session, 'get_session_info')['default_task'] == {'name': 'stack_blocks_by_language', 'instruction': None}
        reset = call(session, 'reset_task')
        assert reset['task'] == {'name': 'stack_blocks_by_language', 'instruction': GENERATED}
        assert reset['steps_left'] == 400
        call(session, 'stop_episode')
    finally:
        session.close()


def test_agent_chooses_task_through_supervisor_with_per_task_budget(tmp_path):
    backend = supervisor(tmp_path)
    session = McpSession(backend, output_root=tmp_path/'logs')
    try:
        assert backend.info.episode_budget == 550
        reset = call(session, 'reset_task', {'task': 'push_T'})
        assert reset['task'] == {'name': 'push_T', 'instruction': TASKS['push_T'].instruction}
        assert reset['steps_left'] == 600 and backend.worker.info.episode_budget == 600
        status = call(session, 'get_session_info')
        assert status['max_steps'] == 600 and status['task']['name'] == 'push_T'
        call(session, 'set_gripper', {'arm': 'left', 'state': 'close'})
        call(session, 'stop_episode')
        events = session.journal.read_events()
        assert [e['max_steps'] for e in events if e['kind'] == 'controller_params'] == [600]
        # A plain reset stays on the chosen task.
        assert call(session, 'reset_task')['task']['name'] == 'push_T'
        call(session, 'stop_episode')
        bad = session.call_tool('reset_task', {'task': 'no_such_task'})
        assert bad['isError'] and bad['structuredContent']['error'] == 'invalid_input'
        assert session.status == 'stopped' and backend.worker is None
        reset = call(session, 'reset_task', {'task': 'stack_blocks_by_language'})
        assert reset['task']['instruction'] == GENERATED and reset['steps_left'] == 400
        call(session, 'stop_episode')
        summary = json.loads((session.journal.dir/'summary.json').read_text(encoding='utf-8'))
        assert summary['task'] == {'name': 'stack_blocks_by_language', 'instruction': GENERATED}
    finally:
        session.close()
    assert backend.worker is None


def test_locked_server_keeps_the_configured_task(tmp_path):
    backend = supervisor(tmp_path, task='align_blocks')
    session = McpSession(backend, output_root=tmp_path/'logs', task_locked=True)
    try:
        refused = session.call_tool('reset_task', {'task': 'push_T'})
        assert refused['isError'] and refused['structuredContent']['error'] == 'task_locked'
        assert backend.worker is None and session.status == 'idle'
        reset = call(session, 'reset_task')
        assert reset['task']['name'] == 'align_blocks' and reset['steps_left'] == 200
        call(session, 'stop_episode')
    finally:
        session.close()


def test_operator_budget_overrides_every_task_and_is_audited(tmp_path):
    backend = supervisor(tmp_path, episode_budget=10000)
    session = McpSession(backend, output_root=tmp_path/'logs')
    try:
        assert backend.info.episode_budget == 10000
        assert call(session, 'reset_task', {'task': 'align_blocks'})['steps_left'] == 10000
        call(session, 'stop_episode')
    finally:
        session.close()
    for name, override in (('align_blocks', None), ('align_blocks', 10000), ('fasten_screws', None)):
        driver = IsaacDriver(Settings(source='s', python='p', output='o', episode_budget=override), name)
        native = TASKS[name].step_limit
        assert driver.budget == (override or native) and driver.native_budget == native
        driver.env = SimpleNamespace(step_lim=native, take_action_cnt=[0])
        driver._apply_episode_budget()
        assert driver.env.step_lim == driver.budget
    driver = IsaacDriver(Settings(source='s', python='p', output='o'), 'fasten_screws')
    driver.env = SimpleNamespace(step_lim=550, take_action_cnt=[0])
    with pytest.raises(RuntimeError, match='step limit'):
        driver._apply_episode_budget()


@pytest.mark.parametrize('override', [None, 1000])
def test_native_audit_per_task_and_scripted_support_arm_is_not_replayed(tmp_path, override):
    spec = importlib.util.spec_from_file_location('official_writer', TESTS/'fixtures/robodojo_run_eval.py')
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)

    class Env:
        run_eval = oracle.run_eval
        interact = True
        def run_reward(self): raise AssertionError('predicates must not be registered twice')
        def get_score(self): raise AssertionError('scores must not be registered twice')
        def eval_one_episode(self): raise AssertionError('actions must not be repeated')
        def query_support_arm_traj(self, env_idx): raise AssertionError('support arm was queued at reset')
        def get_running_env_idx_list(self): return [] if self.end_flag[0] else [0]
        def save_video(self, *args): pass
        def _abort_video_writers(self): pass
        def persist_resume_manifest(self): pass
    env = Env()
    env.end_flag = [False]; env.success = [True]; env.unstable_envs = set(); env.unstable_nums = 0
    env.reward_manager = SimpleNamespace(get_reward=lambda **kwargs: [0.], get_score=lambda: [20.])
    env.eval_batch = False; env.episode_nums = 1; env.success_nums = env.fail_nums = 0; env.total_score = 0
    env.env_seeds = [0]; env.save_dir = str(tmp_path); env.eval_result = {'details': {}}
    driver = IsaacDriver(Settings(source='s', python='p', output=str(tmp_path), episode_budget=override), 'make_kong')
    driver.env = env
    audit = driver.finish()
    assert audit['native_episode_budget'] == 600 and audit['episode_budget'] == (override or 600)
    assert audit['budget_overridden'] is (override is not None)
    with pytest.raises(AssertionError):
        env.query_support_arm_traj(env_idx=0)


def test_remote_handshake_carries_catalogue_budget_and_generated_instruction(tmp_path):
    # MCP -> P2 service -> supervisor -> worker. The default task's budget (200) is smaller
    # than the chosen task's, and one call may use more than 200 steps.
    backend = supervisor(tmp_path, task='align_blocks')
    with service(backend) as (remote, _):
        assert remote.info.episode_budget == 200
        assert 'task' in remote.catalogue.texts().reset_properties
        assert [row['name'] for row in remote.catalogue.listing({})['tasks']] == list(TASK_NAMES)
        session = McpSession(remote, output_root=tmp_path/'logs', max_steps_per_call=300)
        try:
            assert call(session, 'get_session_info')['default_task']['name'] == 'align_blocks'
            reset = call(session, 'reset_task', {'task': 'classify_objects_by_language'})
            assert reset['task'] == {'name': 'classify_objects_by_language', 'instruction': GENERATED}
            assert reset['steps_left'] == 1100
            # The call allowance (300) exceeds the default task's whole budget.
            call(session, 'move_relative', {'arm': 'left', 'delta_xyz': [0, 0, .02]})
            call(session, 'stop_episode')
            summary = json.loads((session.journal.dir/'summary.json').read_text(encoding='utf-8'))
            assert summary['task']['instruction'] == GENERATED
        finally:
            session.close()
    assert backend.worker is None


def test_serve_accepts_a_call_allowance_up_to_the_chosen_tasks_budget(tmp_path):
    backend = supervisor(tmp_path, task='align_blocks', immobile=True)
    with service(backend) as (remote, _):
        task, variant = remote.catalogue.resolve({'task': 'fasten_screws'}, locked=False)
        remote.catalogue.commit(task, variant)
        remote.reset(task, variant)
        report = remote.set_gripper('left', True, max_steps=1500)
        assert report.executed_steps == 1
        with pytest.raises(McpToolError):
            remote.set_gripper('left', True, max_steps=1901)
        remote.end_episode()


def test_no_evaluator_field_reaches_the_agent_on_any_task_kind(tmp_path):
    backend = supervisor(tmp_path, final_success=True)
    session = McpSession(backend, output_root=tmp_path/'logs', max_steps_per_call=10)
    results = [session.call_tool('get_session_info'), session.call_tool('list_tasks')]
    try:
        for name in ('stack_blocks', 'pour_by_language', 'make_kong'):
            for tool, args in [('reset_task', {'task': name}), ('observe', {}),
                               ('move_relative', {'arm': 'left', 'delta_xyz': [0, 0, .02]}),
                               ('set_gripper', {'arm': 'right', 'state': 'close'}),
                               ('get_session_info', {}), ('stop_episode', {})]:
                results.append(session.call_tool(tool, args))
    finally:
        session.close()
    assert all(not r.get('isError') for r in results), [r for r in results if r.get('isError')]
    blob = json.dumps([session.tools()] + [r['structuredContent'] for r in results], ensure_ascii=False).lower()
    for term in ('success', 'privileged') + tuple(backend.info.forbidden_terms):
        assert term not in blob, term


def test_task_wire_format_keeps_ordinary_tasks_unchanged():
    plain = TaskSpec(suite='libero_spatial', task_index=0, name='t', instruction='i', n_init_states=5)
    assert 'episode_budget' not in pack_task(plain) and unpack_task(pack_task(plain)) == plain
    task = TaskCatalogue().resolve({'task': 'push_T'}, locked=False)[0]
    assert unpack_task(json.loads(json.dumps(pack_task(task)))) == task
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            unpack_task(dict(pack_task(task), episode_budget=bad))


# -- preflight / source ---------------------------------------------------------------


def fake_source(root, names=TASK_NAMES):
    """Minimal files with the shape the preflight parser reads."""
    overrides = ['tasks:']
    for name in TASK_NAMES:
        config = root/'task/RoboDojo/config'/(name + '.yml')
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text('Rigid: []\n')
        if name in SUPPORT_ARM:
            overrides += ['  %s:' % name, '    robot_config: "dual_x5_and_franka_competition"']
        if name in CONVEYOR_TASKS:
            overrides += ['  %s:' % name, '    scene_config: "conveyor"']
    (root/'task/RoboDojo/config/_task.yml').write_text(
        'common:\n  scene_config: "default"\n\n' + '\n'.join(overrides) + '\n')
    for name in names:
        task = TASKS[name]
        module = root/'task/RoboDojo/tasks'/(name + '.py')
        module.parent.mkdir(parents=True, exist_ok=True)
        body = ('return [%r]' % task.instruction if task.instruction
                else 'return [f"stack {self.color}"]')
        module.write_text('class %s:\n    def __init__(self):\n        self.step_lim = %d\n%s'
                          '    def gen_instruction(self, env_idx):\n        %s\n'
                          % (name, task.step_limit, '        self.interact = True\n' if task.support_arm else '', body))
    for path in COMMON_FILES + ('env_cfg/robot/dual_x5_and_franka_competition.yml', 'env_cfg/scene/conveyor.yml'):
        (root/path).parent.mkdir(parents=True, exist_ok=True)
        (root/path).touch()
    return root


def test_preflight_checks_every_task_against_the_table(tmp_path):
    source = fake_source(tmp_path)
    assert preflight.check_tasks(source) == list(TASK_NAMES)
    module = source/'task/RoboDojo/tasks/push_T.py'
    module.write_text(module.read_text().replace('600', '601'))
    with pytest.raises(RuntimeError, match='push_T'):
        preflight.check_tasks(source)
    assert preflight.check_tasks(source, ['stack_blocks']) == ['stack_blocks']
    (source/'env_cfg/robot/dual_x5_and_franka_competition.yml').unlink()
    with pytest.raises(RuntimeError, match='Missing'):
        preflight.check_tasks(source, ['make_kong'])
    (source/'task/RoboDojo/config/extra_task.yml').write_text('')
    with pytest.raises(RuntimeError, match='task list'):
        preflight.check_tasks(source, ['stack_blocks'])


def test_preflight_rejects_changed_task_settings(tmp_path):
    source = fake_source(tmp_path)
    settings = source/'task/RoboDojo/config/_task.yml'
    settings.write_text(settings.read_text() + '  push_T:\n    camera_config: "other"\n')
    with pytest.raises(RuntimeError, match='push_T'):
        preflight.check_tasks(source)


def test_archive_must_cover_the_selected_task_files(tmp_path):
    source = fake_source(tmp_path, names=('stack_blocks',))
    covered = COMMON_FILES + task_files('stack_blocks')
    (source/'.embodify-source-manifest.json').write_text(json.dumps({
        'revisions': {'.': REVISION, **SUBMODULES},
        'sha256': {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in covered}}))
    assert verify_source(source, covered) == source.resolve()
    with pytest.raises(RuntimeError, match='push_T'):
        verify_source(source, COMMON_FILES + task_files('push_T'))


@pytest.mark.skipif(not os.environ.get('EMBODIFY_ROBODOJO_SOURCE'),
                    reason='set EMBODIFY_ROBODOJO_SOURCE to a RoboDojo checkout at the pinned revision')
def test_table_matches_pinned_upstream_source():
    assert preflight.check_tasks(Path(os.environ['EMBODIFY_ROBODOJO_SOURCE'])) == list(TASK_NAMES)
