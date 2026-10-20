"""Additive startup-only source gate; retain every391 numerical obligation."""
import hashlib
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
PREVIOUS = P.parent / 'tree-parent4-current-M120-target-protocol-v391'
PARENT_FREEZE = 'a9a4bfaa81f9bc8f95a75a533612fb1bd8b8202638a6f65c6d35b8185fe8fa98'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(expected=None):
    expected = expected or json.loads((P / 'SOURCE395_RECEIPT.json').read_text())['freeze_sha256']
    assert sha(P / 'FREEZE395.json') == expected
    value = json.loads((P / 'FREEZE395.json').read_text())
    assert value['parent391_freeze_sha256'] == sha(PREVIOUS / 'FREEZE391.json') == PARENT_FREEZE
    for name, digest in value['files'].items():
        assert sha(name) == digest, name
    host = json.loads((P / 'HOST395_STARTUP_RESULT.json').read_text())
    assert host['status'].startswith('PASS_HOST395') and host['NPU_used'] is False
    sys.path.insert(0, str(PREVIOUS))
    from check391 import ready as original
    original(PARENT_FREEZE)
    return value


if __name__ == '__main__':
    ready()
