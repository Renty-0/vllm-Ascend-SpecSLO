"""Complete pinned459 closure plus the new full-serving implementation."""
from pathlib import Path
import hashlib,json,os
from host_env460 import setup
setup()
P=Path(__file__).resolve().parent;B=P.parent
PARENT_FREEZE='74ffbd335bd6223f6e318dfb0de864ecf11f34e41324820426f638dc0810c4fb'
PARENT_ARCHIVE='f00ccf518eb842ca8b69f3dd5cae638a304af7e2fc47fd1f816c565fb4240f88'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ready(expected=None):
    expected=expected or os.environ.get('SPECSLO_V460_ROOT_FREEZE_SHA')
    if not expected or sha(P/'FREEZE460.json')!=expected:raise RuntimeError('ROOT460 explicit exact own source pin')
    value=json.loads((P/'FREEZE460.json').read_text())
    for path,wanted in value['files'].items():
        if sha(path)!=wanted:raise RuntimeError('ROOT460 changed own source '+path)
    import check459
    check459.ready(PARENT_FREEZE)
    if sha(check459.P/'SOURCE459.tar.gz')!=PARENT_ARCHIVE:raise RuntimeError('ROOT460 original459 archive pin')
    for name in('HOST460_REFERENCE_LOOP_RESULT.json','HOST460_COMPLETE_INTERFACE_RESULT.json'):
        result=json.loads((P/name).read_text())
        if not result['status'].startswith('PASS_HOST460')or result['NPU_used']is not False:
            raise RuntimeError('ROOT460 HOST prerequisite '+name)
    return value
if __name__=='__main__':ready()
