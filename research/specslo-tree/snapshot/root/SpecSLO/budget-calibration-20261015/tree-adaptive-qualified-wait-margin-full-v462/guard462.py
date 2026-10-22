"""All original full ownership limits, with declared21600s qualification bound."""
import argparse,fcntl,hashlib,json,re,tarfile
from pathlib import Path
from check462 import P,B,BASE,PARENT_FREEZE,PARENT_ARCHIVE,ready,sha
import guard461 as parent
from projection425 import project_function
DONOR,ACTION,LOCK=parent.DONOR,parent.ACTION,parent.LOCK
MAX_SECONDS=21600;MAX_TAG_BYTES=256*1024**2;MAX_AGGREGATE_BYTES=3584*1024**2
EXPECTED_FREEZE=EXPECTED_ARCHIVE=None
def prepared():
    parent.EXPECTED_FREEZE=PARENT_FREEZE;parent.EXPECTED_ARCHIVE=PARENT_ARCHIVE
    _,files,previous=parent.prepared();own=ready(EXPECTED_FREEZE)
    extra=dict(own['files']);extra[str(P/'FREEZE462.json')]=EXPECTED_FREEZE
    if sha(P/'SOURCE462.tar.gz')!=EXPECTED_ARCHIVE:raise RuntimeError('ROOT462 exact current archive')
    with tarfile.open(P/'SOURCE462.tar.gz','r:gz')as archive:
        members=archive.getmembers()
        if len(members)!=len(extra)or{m.name for m in members}!={p.lstrip('/')for p in extra}:raise RuntimeError('ROOT462 archive inventory')
        for member in members:
            if not member.isfile()or hashlib.sha256(archive.extractfile(member).read()).hexdigest()!=extra['/'+member.name]:raise RuntimeError('ROOT462 archive bytes')
    for path,wanted in extra.items():
        if path in files and files[path]!=wanted:raise RuntimeError('ROOT462 identical inherited source')
        files[path]=wanted
    return own,files,dict(parent461=previous,own_freeze_sha256=EXPECTED_FREEZE,own_archive_sha256=EXPECTED_ARCHIVE,
                         qualification_only_wait_seconds=21600,all_numerical_worker_code_unchanged=True,NN_permission_inherited=False)
def storage(tag=None):
    value=parent.storage(tag);value['aggregate_private_bytes']+=parent.parent.base.donor_guard().bytes_under(P)
    if value['aggregate_private_bytes']>MAX_AGGREGATE_BYTES:raise RuntimeError('ROOT462 original private aggregate bound')
    return value
def launch_argv(tag):
    parent.EXPECTED_FREEZE=PARENT_FREEZE;parent.EXPECTED_ARCHIVE=PARENT_ARCHIVE
    args=parent.launch_argv(tag);old=str(BASE/'launch461.sh')
    if args.count(old)!=1:raise RuntimeError('ROOT462 one actual full launch')
    args=[str(P/'launch462.sh')if value==old else value for value in args]
    first=args.index('/bin/bash');args.insert(first,'SPECSLO_V462_ROOT_FREEZE_SHA='+EXPECTED_FREEZE);return args
def actual_audit(directory):
    body=project_function(parent.actual_audit,(("value['qualification_receive_timeout_seconds']==18000","value['qualification_receive_timeout_seconds']==21600"),))
    result=body(directory)
    completed=json.loads((Path(directory)/'completed.json').read_text())
    if completed['qualification_rpc_wait']['outer_supervisor_total_limit_seconds']!=21600:raise RuntimeError('ROOT462 actual declared HOST-only wait bound')
    return result
def execution_namespace():
    factory=project_function(parent.execution_namespace,(
        ('ROOT.V461.retained-CANN-adaptive-full-own-AR-supervisor.v1','ROOT.V462.qualification-margin-adaptive-full-own-AR-supervisor.v1'),
        ('ROOT461 supervisor received','ROOT462 supervisor received'),
    ),dict(P=P,prepared=prepared,storage=storage,launch_argv=launch_argv,actual_audit=actual_audit))
    ns=factory();ns['MAX_SECONDS']=MAX_SECONDS
    ns['SCOPE'].update(qualification_only_wait_seconds=21600,original_worker_and_production_timeouts_unchanged=True)
    if ns['execute_locked'].__globals__ is not ns:raise RuntimeError('ROOT462 actual owned callbacks')
    return ns
def main():
    global EXPECTED_FREEZE,EXPECTED_ARCHIVE
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--expected-freeze-sha256',required=True);parser.add_argument('--expected-archive-sha256',required=True);args=parser.parse_args()
    EXPECTED_FREEZE=args.expected_freeze_sha256;EXPECTED_ARCHIVE=args.expected_archive_sha256
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)or any(not re.fullmatch('[a-f0-9]{64}',value)for value in(EXPECTED_FREEZE,EXPECTED_ARCHIVE)):raise RuntimeError('ROOT462 safe tag and explicit source pins')
    prepared();ns=execution_namespace()
    with LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ns['execute_locked'](args.tag)
if __name__=='__main__':main()
