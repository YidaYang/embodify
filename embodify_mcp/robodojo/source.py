"""Verify pinned Git checkouts or a locally prepared source archive."""
import hashlib
import json
import subprocess
from pathlib import Path
from .config import REVISION, SUBMODULES


def verify_source(source, required=()):
    """`required` files must exist and, in an archive, be covered by its hash manifest."""
    source = Path(source).resolve()
    pins = {'.': REVISION, **SUBMODULES}
    manifest = source / '.embodify-source-manifest.json'
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding='utf-8'))
        if data['revisions'] != pins or not data['sha256']:
            raise RuntimeError('Unsupported source archive')
        for name in required:
            if name not in data['sha256']:
                raise RuntimeError('Source archive does not cover: ' + name)
        for name, expected in data['sha256'].items():
            path = (source / name).resolve()
            if source not in path.parents or not path.is_file():
                raise RuntimeError('Missing source file: ' + name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('Changed source file: ' + name)
    else:
        for directory, revision in pins.items():
            repo = str(source / directory)
            actual = subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()
            if actual != revision:
                raise RuntimeError('Unsupported upstream revision: ' + directory)
            changed = subprocess.check_output(['git','-C',repo,'diff','HEAD','--name-only','--ignore-submodules=all'],text=True)
            if changed.strip():
                raise RuntimeError('Modified upstream source: ' + directory)
        for name in required:
            if not (source / name).is_file():
                raise RuntimeError('Missing ' + name)
    return source
