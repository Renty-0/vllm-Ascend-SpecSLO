"""Own finite target adapter pin plus all original386/199 source obligations."""
import hashlib
import json
import os
from pathlib import Path
import sys
P=Path(__file__).parent;B=P.parent
sys.path.insert(0,str(B/'tree-parent4-child4-original-bounded-v386'))
import check386


def ready(expected=None):
    expected=expected or os.environ.get('SPECSLO_V391_ROOT_FREEZE_SHA')
    manifest=P/'FREEZE391.json'
    if not expected or hashlib.sha256(manifest.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('ROOT391 explicit frozen source pin required')
    value=json.loads(manifest.read_text())
    for path,digest in value['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('ROOT391 immutable source changed '+path)
    for path,prefix in [(P/'HOST391_RESULT.json','PASS_HOST391'),
            (B/'tree-parent4-best376-target-binding-host-v390/HOST390_RESULT.json','PASS_HOST390')]:
        host=json.loads(path.read_text())
        if not host['status'].startswith(prefix)or host['NPU_used']is not False:
            raise RuntimeError('ROOT391 current HOST evidence required')
    physical=json.loads((P/'HOST391_PHYSICAL_RESULT.json').read_text())
    if not physical['status'].startswith('PASS_HOST391_REAL_OBSERVER')or physical['NPU_used']is not False:
        raise RuntimeError('ROOT391 actual physical metadata observer HOST evidence required')
    early=json.loads((B/'tree-parent4-frontier-observer-before-capture-host-v392/HOST392_RESULT.json').read_text())
    if not early['status'].startswith('PASS_HOST392_ACTUAL_ORIGINAL_OBSERVER')or early['NPU_used']is not False:
        raise RuntimeError('ROOT391 early observer actual HOST regression required')
    check386.ready()
    return value


def require_ready(expected_identity=None):
    ready()
    return check386.require_ready(expected_identity)


if __name__=='__main__':ready()
