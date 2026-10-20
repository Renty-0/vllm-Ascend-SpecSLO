"""Own finite current-M120 protocol supervisor; original cleanup and limits."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import sys
HERE=Path(__file__).parent;B=HERE.parent
sys.path.insert(0,str(B/'tree-parent4-atomic-output-storage-guard-v389'))
import guard389 as parent
PARENT_FREEZE='e233543cad3a82dc6defce6f17f7d32c30a6e96a58a39c9ec12a7298a49cde29'
PARENT_ARCHIVE='0527ec6a75a5f0b006bbb52aaa8b01c975c34594ae2f5fed8277b6d06844203d'
EXPECTED_FREEZE=EXPECTED_ARCHIVE=None
require=parent.require


def prepared():
    parent.EXPECTED_FREEZE=PARENT_FREEZE;parent.EXPECTED_ARCHIVE=PARENT_ARCHIVE
    _,sources,previous=parent.prepared()
    own,extra=parent.parent.parent.checked_archive(HERE,391,EXPECTED_FREEZE,EXPECTED_ARCHIVE)
    require(own['parent389_freeze_sha256']==PARENT_FREEZE and own['parent389_archive_sha256']==PARENT_ARCHIVE,'bound actual original389 runtime source')
    require(not set(sources)&set(extra),'disjoint391 finite target source')
    sources.update(extra)
    return own,sources,dict(parent=previous,own_freeze_sha256=EXPECTED_FREEZE,
        own_archive_sha256=EXPECTED_ARCHIVE,source_members=len(sources),model_qualification_inherited=False,
        target_delta='finite original199 protocol target delegate uses actual best376 M120 backend')


def launch_argv(tag):
    args=parent.parent.launch_argv(tag)
    before=str(B/'tree-parent4-owned-exit-path-v388/startup388.sh')
    require(args.count(before)==1,'one actual original388 startup')
    args=[str(HERE/'startup391.sh')if a==before else a for a in args]
    offset=args.index('/bin/bash')
    args.insert(offset,'SPECSLO_V391_ROOT_FREEZE_SHA='+EXPECTED_FREEZE)
    return args


def storage(tag=None):
    value=parent.storage(tag)
    value['aggregate_private_bytes']+=parent.bytes_under(HERE)+parent.bytes_under(B/'tree-parent4-best376-target-binding-host-v390')
    require(value['aggregate_private_bytes']<=parent.parent.parent.original.MAX_AGGREGATE_BYTES,'unchanged aggregate private storage limit')
    return value


def actual_audit(directory):
    sys.path.insert(0,str(HERE));sys.path.insert(0,str(B/'tree-parent4-best376-target-binding-host-v390'))
    sys.path.insert(0,str(parent.DONOR))
    from audit391 import audit
    result=audit(directory);done=json.loads((Path(directory)/'completed.json').read_text())
    require(done['audit']==result,'actual391 child and supervisor finite audits agree')
    require(done['replies']==[['calibrated',i,dict(bounded199=True,production_permission=False)]for i in range(4)],'actual original four worker replies')
    require(json.loads((Path(directory)/'worker-exit.json').read_text())==dict(closed=True,all_exited=True),'actual owned worker exit')
    return result,'finite_parent4_current_M120_protocol_no_production',dict(hidden_KV_recomputed_by_HOST=False)


def execution_namespace():
    namespace=parent.execution_namespace()
    namespace.update(prepared=prepared,launch_argv=launch_argv,storage=storage,actual_audit=actual_audit,G=HERE)
    namespace['SCOPE']=dict(namespace['SCOPE'],current_M120_target_requested=True,
        original_best376_draft_claimed=False)
    require(namespace['execute_locked'].__globals__ is namespace,'actual391 supervisor callbacks')
    return namespace


def main():
    global EXPECTED_FREEZE,EXPECTED_ARCHIVE
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--expected-freeze-sha256',required=True);parser.add_argument('--expected-archive-sha256',required=True)
    args=parser.parse_args();EXPECTED_FREEZE,EXPECTED_ARCHIVE=args.expected_freeze_sha256,args.expected_archive_sha256
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)is not None and all(re.fullmatch('[0-9a-f]{64}',x)for x in [EXPECTED_FREEZE,EXPECTED_ARCHIVE]),'safe explicit391 tag/pins')
    prepared();namespace=execution_namespace()
    with parent.parent.parent.original.LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);namespace['execute_locked'](args.tag)


if __name__=='__main__':main()
