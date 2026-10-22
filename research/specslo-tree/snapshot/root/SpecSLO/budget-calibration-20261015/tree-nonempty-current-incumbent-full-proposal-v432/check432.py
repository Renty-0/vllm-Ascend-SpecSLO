"""HOST proposal source gate; it grants no model or production permission."""
import hashlib,json,tarfile
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
B=P.parent
PARENTS={
    'tree-parent4-preflight-component-lifetime-v433':(433,'67560bc431ed7fd44aad19c49e060e4049227f128570a43d433dc5e491330a27','a75a6cb9553036bbfd56d15a4019baf8bb17fd7f95f73b300f2b70eece7ac730'),
    'tree-nonempty-wholeprompt-refill-resource-host-v426':(426,'7753e8949156edfde65a7ce7c7b558b9843f3cbd438b18ca341f047678073600','01319b786998c9d6184775e8f8e438500545898322b196f4f1580303955e5869'),
    'tree-nonempty-prefill-incumbent-KV-protocol-host-v427':(427,'77a9c38924ffa43b578b0efda077a15ddc7f5120cc7794450941aaca2ef39b36','740d69e47f531b2264b5e4da85a58d937fdee06af815bf5fe6721461e24e2f83'),
    'tree-nonempty-refill-accounting-witness-host-v429':(429,'8a0eacfe0088ef64f31448fefb07e4ae1d65ab33fe49d3486743dde84a2f866a','393b0d77b715891884b14fae27607b3abe8ea84edde029c90675da9e25343e86'),
    'tree-nonempty-refill-current-proof-host-v431':(431,'e02fb5f54b9e92687cfdeba2ab92e7f53ca33a6844e4d5dc2c6e940f027dad28','0ebf7517148ca1eb60910f8ab100ea77afa459e6a2947f1e2506bc06fb1ce7af'),
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_archive(directory,number,freeze_sha,archive_sha):
    freeze=directory/f'FREEZE{number}.json';archive=directory/f'SOURCE{number}.tar.gz'
    if sha(freeze)!=freeze_sha or sha(archive)!=archive_sha:
        raise RuntimeError('ROOT432 exact source proposal parent '+str(directory))
    value=json.loads(freeze.read_text());files=dict(value['files']);files[str(freeze)]=freeze_sha
    for path,digest in files.items():
        if sha(path)!=digest:raise RuntimeError('ROOT432 retained source changed '+path)
    with tarfile.open(archive,'r:gz')as saved:
        members=saved.getmembers()
        if len(members)!=len(files)or {m.name for m in members}!={p.lstrip('/')for p in files}:
            raise RuntimeError('ROOT432 complete source-only archive inventory')
        for member in members:
            if not(member.isfile()and hashlib.sha256(saved.extractfile(member).read()).hexdigest()==files['/'+member.name]):
                raise RuntimeError('ROOT432 archived source bytes')
    return files


def ready(expected=None):
    receipt=json.loads((P/'SOURCE432_RECEIPT.json').read_text())
    expected=expected or receipt['freeze_sha256']
    if sha(P/'FREEZE432.json')!=expected:raise RuntimeError('ROOT432 exact HOST proposal source freeze')
    value=json.loads((P/'FREEZE432.json').read_text())
    for path,digest in value['files'].items():
        if sha(path)!=digest:raise RuntimeError('ROOT432 own frozen source '+path)
    for folder,(n,f,a)in PARENTS.items():checked_archive(B/folder,n,f,a)
    sys.path.insert(0,str(B/'tree-parent4-preflight-component-lifetime-v433'))
    import check433
    check433.ready(PARENTS['tree-parent4-preflight-component-lifetime-v433'][1])
    result=json.loads((P/'HOST432_COMPOSITION_RESULT.json').read_text())
    if not(result['status'].startswith('PASS_HOST432')and result['NPU_used']is False
           and result['original218_close_after_all75_and_extra427_before_E4_issue']is True):
        raise RuntimeError('ROOT432 actual HOST full composition prerequisite')
    return value
