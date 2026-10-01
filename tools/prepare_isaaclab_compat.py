"""Build a labelled headless package variant without editing pinned IsaacLab."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError('Compatibility copy must be outside the pinned package')
    setup = (source/'setup.py').read_text(encoding='utf-8')
    requirement = '    "starlette==0.49.1",'
    version = 'version=EXTENSION_TOML_DATA["package"]["version"],'
    if setup.count(requirement) != 1 or setup.count(version) != 1:
        raise RuntimeError('Unexpected upstream packaging; review the compatibility patch')
    shutil.copytree(source, output, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.egg-info', 'build', 'dist'))
    patched = setup.replace(requirement, '    # Embodify headless variant: livestream service is not enabled.')
    patched = patched.replace(version, 'version=EXTENSION_TOML_DATA["package"]["version"] + "+embodify.headless",')
    (output/'setup.py').write_text(patched, encoding='utf-8')
    hashes = {}
    for original in (source/'isaaclab').rglob('*.py'):
        relative = original.relative_to(source)
        data = original.read_bytes()
        if (output/relative).read_bytes() != data:
            raise RuntimeError('Runtime source copy differs: '+str(relative))
        hashes[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    report = {'variant': 'embodify-headless', 'source': str(source),
              'removed_requirement': 'starlette==0.49.1',
              'reason': 'Unused livestream constraint conflicts with pinned Isaac Sim 5.1 FastAPI stack',
              'runtime_python_sha256': hashes}
    (output/'.embodify-compatibility.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    r = prepare(a.source, a.output)
    print('P5_HEADLESS_PACKAGE_READY', len(r['runtime_python_sha256']))
