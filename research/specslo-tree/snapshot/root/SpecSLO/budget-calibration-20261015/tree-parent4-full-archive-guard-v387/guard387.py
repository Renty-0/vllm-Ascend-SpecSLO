"""Own supervisor for the complete386 archive, retaining all original199 actors."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

HERE=Path(__file__).parent;B=HERE.parent
BASE=B/'tree-parent4-child4-original-bounded-v386'
BASE_FREEZE='0909f2ed2a3c5fbd0ef94a3e9122611c48bf4cc4e164afdbd558ce27afbb1b96'
BASE_ARCHIVE='0be545fb5f19bd4a817ec5b6399a0521c0937525f00d57ddc20720765a1bfb49'
sys.path.insert(0,str(BASE))
import guard386 as original
EXPECTED_FREEZE=EXPECTED_ARCHIVE=None


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_archive(folder,version,freeze_sha,archive_sha):
    manifest=folder/f'FREEZE{version}.json';archive=folder/f'SOURCE{version}.tar.gz'
    original.require(sha(manifest)==freeze_sha and sha(archive)==archive_sha,'explicit original archive pins')
    value=json.loads(manifest.read_text());sources=dict(value['files']);sources[str(manifest)]=freeze_sha
    for path,digest in sources.items():original.require(sha(path)==digest,'immutable current source '+path)
    with tarfile.open(archive,'r:gz')as saved:
        members=saved.getmembers()
        original.require(len(members)==len(sources)and {m.name for m in members}=={p.lstrip('/')for p in sources},'exact complete archive members')
        for member in members:
            original.require(member.isfile()and hashlib.sha256(saved.extractfile(member).read()).hexdigest()==sources['/'+member.name],'exact archived source bytes')
    return value,sources


def prepared():
    base,files=checked_archive(BASE,386,BASE_FREEZE,BASE_ARCHIVE)
    own,extra=checked_archive(HERE,387,EXPECTED_FREEZE,EXPECTED_ARCHIVE)
    original.require(not set(extra)&set(files),'disjoint supervisor sources')
    original.require(own['complete386_freeze_sha256']==BASE_FREEZE and own['complete386_archive_sha256']==BASE_ARCHIVE,'bound complete386 runtime')
    files.update(extra)
    return own,files,dict(own_freeze_sha256=EXPECTED_FREEZE,own_archive_sha256=EXPECTED_ARCHIVE,
        complete386_freeze_sha256=BASE_FREEZE,complete386_archive_sha256=BASE_ARCHIVE,
        source_members=len(files),device_qualification_inherited=False,
        runtime_delta='original199 source gate and owned exit only; supervisor handles complete source archive')


def execution_namespace():
    original.EXPECTED_FREEZE=BASE_FREEZE;original.EXPECTED_ARCHIVE=BASE_ARCHIVE
    namespace=original.execution_namespace()
    namespace.update(prepared=prepared,G=HERE)
    if namespace['execute_locked'].__globals__ is not namespace:
        raise RuntimeError('ROOT387 real supervisor callbacks must share updated namespace')
    return namespace


def main():
    global EXPECTED_FREEZE,EXPECTED_ARCHIVE
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',required=True);parser.add_argument('--expected-freeze-sha256',required=True)
    parser.add_argument('--expected-archive-sha256',required=True);args=parser.parse_args()
    EXPECTED_FREEZE,EXPECTED_ARCHIVE=args.expected_freeze_sha256,args.expected_archive_sha256
    original.require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)is not None
        and all(re.fullmatch('[0-9a-f]{64}',s)for s in [EXPECTED_FREEZE,EXPECTED_ARCHIVE]),'safe explicit pins and tag')
    prepared();namespace=execution_namespace()
    with original.LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)


if __name__=='__main__':main()
