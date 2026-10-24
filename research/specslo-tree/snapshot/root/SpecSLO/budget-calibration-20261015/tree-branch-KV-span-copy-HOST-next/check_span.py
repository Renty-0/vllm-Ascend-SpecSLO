"""Explicit prospective source binding, not neural qualification."""
from pathlib import Path
import hashlib
import json
import os
from host_env_span import setup
setup()

P = Path(__file__).resolve().parent
B = P.parent
PARENT = B/'tree-M80-own-geometry-whole-model-v457'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(expected=None, *, include_parent=False):
    expected = expected or os.environ.get('SPECSLO_BRANCH_SPAN_COPY_FREEZE_SHA')
    if not expected or sha(P/'FREEZE_PROBE.json') != expected:
        raise RuntimeError('Explicit independent parallel-fork source freeze required')
    own = json.loads((P/'FREEZE_PROBE.json').read_text())
    for path, wanted in own['files'].items():
        if sha(path) != wanted:
            raise RuntimeError('Changed parallel-fork source: '+path)
    from check457 import ready as original
    pins = json.loads((PARENT/'SOURCE457_RECEIPT.json').read_text())
    if (own['parent_freeze_sha256'], own['parent_archive_sha256']) != (
            pins['freeze_sha256'], pins['archive_sha256']):
        raise RuntimeError('Exact original diagnostic closure required')
    _, files = original(pins['freeze_sha256'], include_parent=True)
    return (own, files) if include_parent else own


if __name__ == '__main__':
    ready()
