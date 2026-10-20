"""Own finite N24 source closure, retaining the unchanged395 parent chain."""
import hashlib
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
PREVIOUS = P.parent / 'tree-parent4-current-M120-import-repair-v395'
PARENT_FREEZE = 'd3ccc5f6547ec6844df3308132a5f759beb23c08bec93a4befe6d585d7716516'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(expected=None):
    expected = expected or json.loads((P / 'SOURCE396_RECEIPT.json').read_text())['freeze_sha256']
    assert sha(P / 'FREEZE396.json') == expected
    value = json.loads((P / 'FREEZE396.json').read_text())
    assert value['parent395_freeze_sha256'] == sha(PREVIOUS / 'FREEZE395.json') == PARENT_FREEZE
    for name, digest in value['files'].items():
        assert sha(name) == digest, name
    host = json.loads((P / 'HOST396_SCOPE_RESULT.json').read_text())
    assert host['status'].startswith('PASS_HOST396') and host['NPU_used'] is False
    sys.path.insert(0, str(PREVIOUS))
    from check395 import ready as original
    original(PARENT_FREEZE)
    return value


def require_ready(expected_identity=None):
    ready()
    from check391 import require_ready as original
    return original(expected_identity)


if __name__ == '__main__':
    ready()
