"""Original full460 supervisor, with preserved inherited CANN Python paths."""
from pathlib import Path
import argparse,fcntl,hashlib,json,re,tarfile
from check461 import P,B,BASE,PARENT_FREEZE,PARENT_ARCHIVE,ready,sha
import guard460 as parent
from projection425 import project_function
DONOR,ACTION,LOCK=parent.DONOR,parent.ACTION,parent.LOCK
MAX_SECONDS,MAX_TAG_BYTES,MAX_AGGREGATE_BYTES=parent.MAX_SECONDS,parent.MAX_TAG_BYTES,parent.MAX_AGGREGATE_BYTES
EXPECTED_FREEZE=EXPECTED_ARCHIVE=None
def prepared():
    parent.EXPECTED_FREEZE=PARENT_FREEZE;parent.EXPECTED_ARCHIVE=PARENT_ARCHIVE
    _,files,previous=parent.prepared();own=ready(EXPECTED_FREEZE)
    extra=dict(own['files']);extra[str(P/'FREEZE461.json')]=EXPECTED_FREEZE
    if sha(P/'SOURCE461.tar.gz')!=EXPECTED_ARCHIVE:raise RuntimeError('ROOT461 exact source archive')
    with tarfile.open(P/'SOURCE461.tar.gz','r:gz')as archive:
        members=archive.getmembers()
        if len(members)!=len(extra)or{m.name for m in members}!={p.lstrip('/')for p in extra}:raise RuntimeError('ROOT461 exact archive inventory')
        for member in members:
            if not member.isfile()or hashlib.sha256(archive.extractfile(member).read()).hexdigest()!=extra['/'+member.name]:raise RuntimeError('ROOT461 exact archive bytes')
    for path,wanted in extra.items():
        if path in files and files[path]!=wanted:raise RuntimeError('ROOT461 identical inherited source')
        files[path]=wanted
    return own,files,dict(parent460=previous,own_freeze_sha256=EXPECTED_FREEZE,own_archive_sha256=EXPECTED_ARCHIVE,numerical_parent460_unchanged=True,NN_permission_inherited=False)
def storage(tag=None):
    value=parent.storage(tag);value['aggregate_private_bytes']+=parent.base.donor_guard().bytes_under(P)
    if value['aggregate_private_bytes']>MAX_AGGREGATE_BYTES:raise RuntimeError('ROOT461 retained aggregate private bound')
    return value
def launch_argv(tag):
    parent.EXPECTED_FREEZE=PARENT_FREEZE;parent.EXPECTED_ARCHIVE=PARENT_ARCHIVE
    args=parent.launch_argv(tag);old=str(BASE/'launch460.sh')
    if args.count(old)!=1:raise RuntimeError('ROOT461 one actual full launch')
    args=[str(P/'launch461.sh')if value==old else value for value in args]
    first=args.index('/bin/bash');args.insert(first,'SPECSLO_V461_ROOT_FREEZE_SHA='+EXPECTED_FREEZE);return args
actual_audit=parent.actual_audit
def execution_namespace():
    factory=project_function(parent.execution_namespace,(
        ('ROOT.V460.actual-adaptive80-120-full-own-AR-supervisor.v1','ROOT.V461.retained-CANN-adaptive-full-own-AR-supervisor.v1'),
        ('ROOT460 supervisor received','ROOT461 supervisor received'),
    ),dict(P=P,prepared=prepared,storage=storage,launch_argv=launch_argv,actual_audit=actual_audit))
    ns=factory();ns['SCOPE'].update(numerical_parent460_unchanged=True,CANN_Python_paths_preserved=True)
    if ns['execute_locked'].__globals__ is not ns:raise RuntimeError('ROOT461 actual owned supervisor callbacks')
    return ns
def main():
    global EXPECTED_FREEZE,EXPECTED_ARCHIVE
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--expected-freeze-sha256',required=True);parser.add_argument('--expected-archive-sha256',required=True);args=parser.parse_args()
    EXPECTED_FREEZE=args.expected_freeze_sha256;EXPECTED_ARCHIVE=args.expected_archive_sha256
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)or any(not re.fullmatch('[a-f0-9]{64}',value)for value in(EXPECTED_FREEZE,EXPECTED_ARCHIVE)):raise RuntimeError('ROOT461 safe tag and explicit source pins')
    prepared();ns=execution_namespace()
    with LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ns['execute_locked'](args.tag)
if __name__=='__main__':main()
