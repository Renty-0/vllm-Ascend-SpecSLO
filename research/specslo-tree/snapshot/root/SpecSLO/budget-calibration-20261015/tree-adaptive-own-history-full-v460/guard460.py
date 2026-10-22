"""Original full438 ownership/limits with fresh adaptive math and own-AR."""
import argparse,fcntl,hashlib,json,re,tarfile
from pathlib import Path
from host_env460 import setup,source_environment
setup()
from check460 import P,B,ready,sha,PARENT_FREEZE,PARENT_ARCHIVE
import guard459 as parent
import guard438 as full
import guard425 as base
from projection425 import project_function
DONOR,ACTION,LOCK=base.DONOR,base.ACTION,base.LOCK
MAX_SECONDS=18000;MAX_TAG_BYTES=256*1024**2;MAX_AGGREGATE_BYTES=3584*1024**2
EXPECTED_FREEZE=EXPECTED_ARCHIVE=None
def prepared():
    parent.EXPECTED_FREEZE=PARENT_FREEZE;parent.EXPECTED_ARCHIVE=PARENT_ARCHIVE
    _,files,previous=parent.prepared();own=ready(EXPECTED_FREEZE)
    extra=dict(own['files']);extra[str(P/'FREEZE460.json')]=EXPECTED_FREEZE
    if sha(P/'SOURCE460.tar.gz')!=EXPECTED_ARCHIVE:raise RuntimeError('ROOT460 exact complete source archive')
    with tarfile.open(P/'SOURCE460.tar.gz','r:gz')as archive:
        members=archive.getmembers()
        if len(members)!=len(extra)or {m.name for m in members}!={p.lstrip('/')for p in extra}:
            raise RuntimeError('ROOT460 exact archived source inventory')
        for member in members:
            if not member.isfile()or hashlib.sha256(archive.extractfile(member).read()).hexdigest()!=extra['/'+member.name]:
                raise RuntimeError('ROOT460 exact archived source bytes')
    for path,wanted in extra.items():
        if path in files and files[path]!=wanted:raise RuntimeError('ROOT460 identical inherited source')
        files[path]=wanted
    return own,files,dict(parent459=previous,own_freeze_sha256=EXPECTED_FREEZE,own_archive_sha256=EXPECTED_ARCHIVE,
                         all_original_full438_gates_retained=True,current_additional_own_physical_cases=693,
                         global_active_limit=64,device_permission_inherited=False,postfull_own_AR_required=True)
def storage(tag=None):
    value=parent.storage(None);value['actual_tag_bytes']=0 if tag is None else base.donor_guard().bytes_under(DONOR/'actual'/tag)
    value['aggregate_private_bytes']+=base.donor_guard().bytes_under(P)
    # The full run retains the original full-serving256MiB budget; the parent
    # measurement-only supervisor's192MiB limit is not a serving requirement.
    if value['actual_tag_bytes']>MAX_TAG_BYTES or value['aggregate_private_bytes']>MAX_AGGREGATE_BYTES:
        raise RuntimeError('ROOT460 original full private256/3584MiB bounds')
    return value
def launch_argv(tag):
    args=full.launch_argv(tag);old=str(full.P/'launch438.sh')
    if args.count(old)!=1:raise RuntimeError('ROOT460 one original full launch entry')
    args=[str(P/'launch460.sh')if value==old else value for value in args]
    pins=source_environment();pins.pop('PYTHONPATH');pins['SPECSLO_V460_ROOT_FREEZE_SHA']=EXPECTED_FREEZE
    first=args.index('/bin/bash');args[first:first]=[key+'='+value for key,value in sorted(pins.items())]
    return args
def actual_audit(directory):
    actual=project_function(base.actual_audit,(('from audit425 import audit','from audit460 import audit'),))
    result=actual(directory)
    value=base.read(Path(directory)/'completed.json')['qualification_rpc_wait']
    base.require(value['original_parent_timeout_seconds']==6000
                 and value['qualification_receive_timeout_seconds']==18000
                 and value['original_engine_configuration_object_unchanged']is True
                 and value['worker_configuration_changed']is False and value['production_timeout_changed']is False,
                 'actual retained parent-only qualification receiver view')
    return result
def execution_namespace():
    factory=project_function(full.execution_namespace,(
        ('ROOT.V438.best257-uniform-KV-full-own-AR-supervisor.v1','ROOT.V460.actual-adaptive80-120-full-own-AR-supervisor.v1'),
        ('ROOT438 supervisor received','ROOT460 supervisor received'),
    ),dict(P=P,prepared=prepared,storage=storage,launch_argv=launch_argv,actual_audit=actual_audit))
    ns=factory();ns['SCOPE'].update(current_additional_own_physical_cases=693,adaptive_M=[80,120],
                                    original_prompt_prefill_unchanged=True,geometry_matched_independent_own_AR_required=True)
    if ns['execute_locked'].__globals__ is not ns or ns['MAX_SECONDS']!=18000 or ns['MAX_TAG_BYTES']!=256*1024**2:
        raise RuntimeError('ROOT460 actual original full-serving supervision')
    return ns
def main():
    global EXPECTED_FREEZE,EXPECTED_ARCHIVE
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--expected-freeze-sha256',required=True);parser.add_argument('--expected-archive-sha256',required=True);args=parser.parse_args()
    EXPECTED_FREEZE=args.expected_freeze_sha256;EXPECTED_ARCHIVE=args.expected_archive_sha256
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)or any(not re.fullmatch('[a-f0-9]{64}',value)for value in(EXPECTED_FREEZE,EXPECTED_ARCHIVE)):
        raise RuntimeError('ROOT460 safe tag and explicit source pins')
    prepared();ns=execution_namespace()
    with LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ns['execute_locked'](args.tag)
if __name__=='__main__':main()
