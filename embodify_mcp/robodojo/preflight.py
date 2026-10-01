"""Read-only source/assets/dependency checks. Does not initialize Isaac or CUDA."""
import argparse
import ast
import importlib.metadata
import json
import re
import sys
from pathlib import Path
from embodify_mcp.robodojo.config import REVISION, SUBMODULES, ASSET_REVISION
from embodify_mcp.robodojo.source import verify_source
from embodify_mcp.robodojo.tasks import COMMON_FILES, CONVEYOR_TASKS, TASK_NAMES, TASKS, task_files


def read_task_facts(source, name):
    """Step limit, static instruction and support arm, parsed from the task module without importing it.

    The instruction is None when `gen_instruction` builds it per episode (an f-string, or a
    `<label>` placeholder that the description manager fills from the layout).
    """
    tree = ast.parse((Path(source)/'task/RoboDojo/tasks'/(name+'.py')).read_text(encoding='utf-8'))
    limits, interact, generator = [], False, None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id == 'self' and isinstance(node.value, ast.Constant)):
                if target.attr == 'step_lim': limits.append(node.value.value)
                if target.attr == 'interact': interact = interact or node.value.value is True
        if isinstance(node, ast.FunctionDef) and node.name == 'gen_instruction':
            generator = node
    if len(set(limits)) != 1 or type(limits[0]) is not int or generator is None:
        raise RuntimeError('Cannot read task definition: ' + name)
    body = list(ast.walk(generator))
    strings = [n.value for n in body if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    dynamic = any(isinstance(n, ast.JoinedStr) for n in body)
    instruction = None
    if not dynamic and len(strings) == 1 and not re.search(r'<[^<>]+>', strings[0]):
        instruction = strings[0]
    return {'step_limit': limits[0], 'instruction': instruction, 'support_arm': interact}


def read_task_overrides(source):
    """Per-task settings from `_task.yml` (two-level mapping; parsed without a YAML dependency)."""
    overrides, section, task = {}, None, None
    for line in (Path(source)/'task/RoboDojo/config/_task.yml').read_text(encoding='utf-8').splitlines():
        match = re.match(r'^( *)([A-Za-z_]\w*):(.*)$', line.split('#')[0].rstrip())
        if not match: continue
        indent, key, value = len(match.group(1)), match.group(2), match.group(3).strip().strip('"\'')
        if indent == 0:
            section = key
        elif section == 'tasks' and indent == 2:
            task = overrides.setdefault(key, {})
        elif section == 'tasks' and indent == 4 and task is not None:
            task[key] = value
    return overrides


def check_tasks(source, names=TASK_NAMES):
    """Every selected task's files exist and match the pinned table the catalogue advertises."""
    for name in names:
        if name not in TASKS: raise RuntimeError('Unknown RoboDojo task: ' + name)
    offered = sorted(p.stem for p in (Path(source)/'task/RoboDojo/config').glob('*.yml') if not p.stem.startswith('_'))
    if offered != sorted(TASK_NAMES): raise RuntimeError('Upstream task list differs from the pinned catalogue')
    overrides = read_task_overrides(source)
    for name in TASK_NAMES:
        entry = overrides.get(name, {})
        # All tasks keep the three-camera rig; support-arm and conveyor scenes are known per task.
        if ('camera_config' in entry
                or (entry.get('robot_config') == 'dual_x5_and_franka_competition') != TASKS[name].support_arm
                or entry.get('robot_config', 'dual_x5') not in ('dual_x5', 'dual_x5_and_franka_competition')
                or (entry.get('scene_config', 'default') == 'conveyor') != (name in CONVEYOR_TASKS)
                or entry.get('scene_config', 'default') not in ('default', 'conveyor')):
            raise RuntimeError('Task settings differ from the pinned catalogue: ' + name)
    for name in names:
        for path in task_files(name):
            if not (Path(source)/path).is_file(): raise RuntimeError('Missing '+path)
        task = TASKS[name]
        expected = {'step_limit': task.step_limit, 'instruction': task.instruction, 'support_arm': task.support_arm}
        if read_task_facts(source, name) != expected:
            raise RuntimeError('Task definition differs from the pinned catalogue: ' + name)
    return list(names)


def check(source, runtime=False, assets=False, tasks=TASK_NAMES):
    tasks = list(tasks)
    required = sorted(set(COMMON_FILES).union(*(task_files(t) for t in tasks if t in TASKS)))
    source = verify_source(source, required)
    for name in ['scripts/install.sh','XPolicyLab/client_server/ws/model_client.py']:
        if not (source/name).is_file(): raise RuntimeError('Missing '+name)
    report = {'source_verified':True,'revision':REVISION,'submodules':SUBMODULES,
              'tasks_checked':check_tasks(source, tasks),
              'assets_checked':False,'runtime_checked':False,'gpu_validated':False}
    if assets:
        manifest = json.loads((source/'.embodify-assets-manifest.json').read_text())
        if manifest['revision'] != ASSET_REVISION: raise RuntimeError('Asset revision differs')
        entries = manifest['files']
        if not entries: raise RuntimeError('Empty asset manifest')
        import hashlib
        for entry in entries:
            path = (source/entry['rfilename']).resolve()
            if source not in path.parents or not path.is_file(): raise RuntimeError('Missing asset: '+str(path))
            if path.stat().st_size != entry['size']: raise RuntimeError('Asset size mismatch: '+str(path))
            # HF LFS SHA256; Git blobs for small ordinary files.
            h = hashlib.sha256() if entry.get('lfs') else hashlib.sha1()
            if not entry.get('lfs'): h.update(('blob %d\0'%entry['size']).encode())
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
            expected = entry['lfs']['sha256'] if entry.get('lfs') else entry['blobId']
            if h.hexdigest()!=expected: raise RuntimeError('Asset digest mismatch: '+str(path))
        report.update(assets_checked=True,asset_files=len(entries),asset_bytes=sum(e['size'] for e in entries))
    if runtime:
        if sys.version_info[:2] != (3,11): raise RuntimeError('Isaac worker requires Python 3.11')
        expected = {'torch':'2.7.0','isaacsim':'5.1.0','numpy':'1.26.0','warp-lang':'1.11.0'}
        installed = {name:importlib.metadata.version(name) for name in expected}
        from packaging.version import Version
        for name, version in expected.items():
            if Version(installed[name].split('+')[0]) != Version(version): raise RuntimeError('Version differs: '+name)
        report.update(runtime_checked=True,versions=installed)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--runtime',action='store_true')
    parser.add_argument('--assets',action='store_true')
    parser.add_argument('--task',action='append',choices=TASK_NAMES,metavar='NAME',
                        help='Check only this task (repeatable); default: every task')
    parser.add_argument('--output')
    a=parser.parse_args()
    result=json.dumps(check(a.source,a.runtime,a.assets,a.task or TASK_NAMES),indent=2)
    if a.output: Path(a.output).write_text(result+'\n',encoding='utf-8')
    print(result)

if __name__=='__main__': main()
