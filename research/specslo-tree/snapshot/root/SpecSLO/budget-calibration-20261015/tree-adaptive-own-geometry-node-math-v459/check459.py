"""New immutable node protocol, complete immutable457 parent, no NN inheritance."""
from pathlib import Path
import hashlib,json,os
from host_env459 import setup
setup()
P=Path(__file__).resolve().parent;B=P.parent
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ready(expected=None):
    expected=expected or os.environ.get('SPECSLO_V459_ROOT_FREEZE_SHA')
    if not expected or sha(P/'FREEZE459.json')!=expected:
        raise RuntimeError('ROOT459 explicit own source freeze')
    value=json.loads((P/'FREEZE459.json').read_text())
    for path,wanted in value['files'].items():
        if sha(path)!=wanted:
            raise RuntimeError('ROOT459 unchanged own source '+path)
    import check457
    check457.ready('6046ff35cab90a252560b2be0bb757b3395174c8fa5b6e33a4225afbb0966228')
    return value
if __name__=='__main__':ready()
