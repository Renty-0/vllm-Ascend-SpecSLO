"""Own immutable sources plus every original199 HOST/source requirement."""
import hashlib
import json
import os
from pathlib import Path
import sys

P=Path(__file__).parent;B=P.parent
PARENT=B/'tree-bounded-parent4-shared4-eager-diagnostic-v199'
sys.path.insert(0,str(PARENT))


def ready(expected=None):
    expected=expected or os.environ.get('SPECSLO_V386_ROOT_FREEZE_SHA')
    f=P/'FREEZE386.json'
    if expected is None or hashlib.sha256(f.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('ROOT386 explicit source pin required')
    value=json.loads(f.read_text())
    for name,digest in value['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('ROOT386 source changed: '+name)
    for name in ['HOST386_CONSTRUCTION_RESULT.json']:
        v=json.loads((P/name).read_text())
        if not v['status'].startswith('PASS_HOST386')or v['NPU_used'] is not False:
            raise RuntimeError('ROOT386 current HOST construction required')
    return value


def require_ready(expected_identity=None):
    ready()
    from check199 import require_ready as original
    return original(expected_identity)


if __name__=='__main__':ready()
