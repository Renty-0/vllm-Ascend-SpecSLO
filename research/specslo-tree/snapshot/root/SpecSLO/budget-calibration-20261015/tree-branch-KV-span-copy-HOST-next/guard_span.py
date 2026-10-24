"""Future bounded screen on current cards; source frozen separately first."""
import argparse
import fcntl
import hashlib
import json
import re
import tarfile
from types import FunctionType
from host_env_span import setup
setup()
from check_span import P,B,PARENT,sha,ready
import guard457 as parent

DONOR,LOCK=parent.DONOR,parent.LOCK
ACTION='parallel-fork-branch-span-copy-concurrent-cost'
PHYSICAL=(0,1,2,7)
MAX_SECONDS=3600
MAX_TAG_BYTES=192*1024**2
MAX_AGGREGATE_BYTES=3584*1024**2
EXPECTED_FREEZE=EXPECTED_ARCHIVE=None


def parent_pins():
    pins=json.loads((PARENT/'SOURCE457_RECEIPT.json').read_text())
    parent.EXPECTED_FREEZE,parent.EXPECTED_ARCHIVE=pins['freeze_sha256'],pins['archive_sha256']


def prepared():
    parent_pins()
    _,files,prior=parent.prepared()
    own=ready(EXPECTED_FREEZE)
    extra=dict(own['files']);extra[str(P/'FREEZE_PROBE.json')]=EXPECTED_FREEZE
    if sha(P/'SOURCE_PROBE.tar.gz')!=EXPECTED_ARCHIVE:
        raise RuntimeError('Exact new screen archive required')
    with tarfile.open(P/'SOURCE_PROBE.tar.gz','r:gz')as archive:
        members=archive.getmembers()
        if len(members)!=len(extra)or{m.name for m in members}!={path.lstrip('/')for path in extra}:
            raise RuntimeError('Complete source archive inventory required')
        for member in members:
            if not member.isfile()or hashlib.sha256(archive.extractfile(member).read()).hexdigest()!=extra['/'+member.name]:
                raise RuntimeError('Exact archived bytes required')
    extra[str(P/'SOURCE_PROBE.tar.gz')]=EXPECTED_ARCHIVE
    for path,wanted in extra.items():
        if path in files and files[path]!=wanted:raise RuntimeError('Consistent inherited source bytes required')
        files[path]=wanted
    return own,files,dict(parent=prior,own_freeze_sha256=EXPECTED_FREEZE,
        own_archive_sha256=EXPECTED_ARCHIVE,physical_cards=list(PHYSICAL),
        NN_permission_inherited=False,serving_permission=False,global_active_limit=64)


def storage(tag=None):
    value=parent.storage(tag)
    value['aggregate_private_bytes']+=sum(f.stat().st_size for f in P.rglob('*')if f.is_file())
    if value['actual_tag_bytes']>MAX_TAG_BYTES or value['aggregate_private_bytes']>MAX_AGGREGATE_BYTES:
        raise RuntimeError('Bounded192/3584MiB private screen required')
    return value


def launch_argv(tag):
    parent_pins();args=parent.launch_argv(tag)
    for old,new in ((str(PARENT/'startup457.sh'),str(P/'startup_span.sh')),
                    (str(B/'serial-v1/runtime.sh'),str(B/'tree-current-card-rebind-full-v464/environment464.sh'))):
        if args.count(old)!=1:raise RuntimeError('One actual owned launch anchor: '+old)
        args=[new if value==old else value for value in args]
    args.insert(args.index('/bin/bash'),'SPECSLO_BRANCH_SPAN_COPY_FREEZE_SHA='+EXPECTED_FREEZE)
    return args


def replace_constants(function,changes,namespace=None):
    values=list(function.__code__.co_consts)
    for old,new in changes:
        found=[i for i,value in enumerate(values)if type(value)is type(old)and value==old]
        if len(found)!=1:raise RuntimeError('One explicit current-device metadata anchor: '+repr(old))
        values[found[0]]=new
    return FunctionType(function.__code__.replace(co_consts=tuple(values)),
        function.__globals__ if namespace is None else namespace,function.__name__,
        function.__defaults__,function.__closure__)


def bind_helpers(original):
    helpers=original();before=helpers['helper']
    def helper():
        value=before()
        if value.PHYSICAL!=(2,5,6,7):raise RuntimeError('Original isolated device helper required')
        value.PHYSICAL=PHYSICAL
        return value
    helpers['helper']=helper
    helpers['verify']=replace_constants(helpers['verify'],(
        ('serial-v1/runtime.sh','tree-current-card-rebind-full-v464/environment464.sh'),))
    return helpers


def actual_audit(directory):
    from audit_span import audit
    return audit(directory)


def execution_namespace():
    ns=parent.execution_namespace();original=ns['original_helpers']
    ns.update(prepared=prepared,storage=storage,launch_argv=launch_argv,actual_audit=actual_audit,
              original_helpers=lambda:bind_helpers(original),ACTION=ACTION,G=P,
              MAX_SECONDS=MAX_SECONDS,MAX_TAG_BYTES=MAX_TAG_BYTES,MAX_AGGREGATE_BYTES=MAX_AGGREGATE_BYTES)
    ns['execute_locked']=replace_constants(ns['execute_locked'],(
        ('ROOT.V457.own80-whole-model-supervisor.v1','ROOT.parallel-fork-span-copy-concurrent-cost-current0127-supervisor.v1'),
        ((2,5,6,7),PHYSICAL),
        ('true M80/S16 using unchanged original profile versus independent ordinary M80 full-model computation and original M120; explicit cross-M120 observations retained; all128 query KV and post-timing checks; no new serving or AR permission',
         'original21 genuine coupled proofs plus seven paired concurrent API timing geometries and14 postcost independent target-node proofs; micro cost only, no serving Goodput'),
    ),ns)
    ns['SCOPE']=dict(current_physical_cards=list(PHYSICAL),finite_cases=21,
                    global_active_limit=64,tree_budgets=[4,8],physical_target_M=[80,120],
                    genuine_draft_target_state_probe=True,performance_measured=True,
                    no_serving_or_whole_workload_AR_permission=True)
    if ns['execute_locked'].__globals__ is not ns:raise RuntimeError('Actual owned callbacks required')
    return ns


def main():
    global EXPECTED_FREEZE,EXPECTED_ARCHIVE
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--expected-freeze-sha256',required=True);p.add_argument('--expected-archive-sha256',required=True)
    args=p.parse_args();EXPECTED_FREEZE=args.expected_freeze_sha256;EXPECTED_ARCHIVE=args.expected_archive_sha256
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)or any(not re.fullmatch('[a-f0-9]{64}',v)for v in(EXPECTED_FREEZE,EXPECTED_ARCHIVE)):
        raise RuntimeError('Fresh safe tag and explicit source pins required')
    with LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepared();execution_namespace()['execute_locked'](args.tag)


if __name__=='__main__':main()
