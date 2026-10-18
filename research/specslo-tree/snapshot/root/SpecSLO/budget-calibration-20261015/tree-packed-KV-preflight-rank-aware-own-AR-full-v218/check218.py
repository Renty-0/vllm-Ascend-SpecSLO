"""ROOT218 source gate; no Torch, device query or model permission on import."""
import hashlib,json,os,tarfile
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
DONOR=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'


def require(value,message):
    if value is not True:raise RuntimeError('ROOT218: '+message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def ready(expected=None):
    expected=expected or os.environ.get('SPECSLO_V218_ROOT_FREEZE_SHA')
    require(isinstance(expected,str)and len(expected)==64 and sha(P/'FREEZE218.json')==expected,'explicit own source freeze')
    value=read(P/'FREEZE218.json')
    for path,digest in value['files'].items():require(sha(path)==digest,'exact source '+path)
    return value


if __name__=='__main__':ready()
