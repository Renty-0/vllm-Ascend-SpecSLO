"""Owned full lifecycle; deepest audit uses470 and retains the original tuple ABI."""
import argparse
import fcntl
import hashlib
import json
import re
import tarfile
from pathlib import Path
from check470 import P,BASE,PARENT_FREEZE,PARENT_ARCHIVE,PHYSICAL,ready,sha
import guard466 as parent
import guard425 as original_full
from projection425 import project_function

DONOR,ACTION,LOCK=parent.DONOR,parent.ACTION,parent.LOCK
MAX_SECONDS=parent.MAX_SECONDS
MAX_TAG_BYTES=parent.MAX_TAG_BYTES
MAX_AGGREGATE_BYTES=parent.MAX_AGGREGATE_BYTES
EXPECTED_FREEZE=EXPECTED_ARCHIVE=None


def prepared():
    parent.EXPECTED_FREEZE,parent.EXPECTED_ARCHIVE=PARENT_FREEZE,PARENT_ARCHIVE
    _,files,previous=parent.prepared()
    own=ready(EXPECTED_FREEZE)
    extra=dict(own['files']);extra[str(P/'FREEZE470.json')]=EXPECTED_FREEZE
    if sha(P/'SOURCE470.tar.gz')!=EXPECTED_ARCHIVE:
        raise RuntimeError('Exact470 source archive required')
    with tarfile.open(P/'SOURCE470.tar.gz','r:gz')as archive:
        members=archive.getmembers()
        if len(members)!=len(extra)or{m.name for m in members}!={path.lstrip('/')for path in extra}:
            raise RuntimeError('Exact470 archive inventory required')
        for member in members:
            if not member.isfile()or hashlib.sha256(archive.extractfile(member).read()).hexdigest()!=extra['/'+member.name]:
                raise RuntimeError('Exact470 archive bytes required')
    extra[str(P/'SOURCE470.tar.gz')]=EXPECTED_ARCHIVE
    for path,wanted in extra.items():
        if path in files and files[path]!=wanted:raise RuntimeError('Identical inherited source bytes required')
        files[path]=wanted
    return own,files,dict(parent466=previous,own_freeze_sha256=EXPECTED_FREEZE,
        own_archive_sha256=EXPECTED_ARCHIVE,physical_cards=list(PHYSICAL),global_active_limit=64,
        maximum_empty_prefill_cohort=48,maximum_nonempty_prefill_cohort=32,
        target_copy_programs=64,original_native_tree_update_method_unchanged=True,
        original_model_weights_admission_and_clocks_unchanged=True,
        current_deepest_auditor_and_original_tuple_ABI=True,NN_permission_inherited=False)


def storage(tag=None):
    value=parent.storage(tag)
    value['aggregate_private_bytes']+=sum(f.stat().st_size for f in P.rglob('*')if f.is_file())
    if value['aggregate_private_bytes']>MAX_AGGREGATE_BYTES:
        raise RuntimeError('Unchanged aggregate private bound exceeded')
    return value


def launch_argv(tag):
    parent.EXPECTED_FREEZE,parent.EXPECTED_ARCHIVE=PARENT_FREEZE,PARENT_ARCHIVE
    args=parent.launch_argv(tag);old=str(BASE/'launch466.sh')
    if args.count(old)!=1:raise RuntimeError('One original466 launch entry required')
    args=[str(P/'launch470.sh')if value==old else value for value in args]
    args.insert(args.index('/bin/bash'),'SPECSLO_V470_ROOT_FREEZE_SHA='+EXPECTED_FREEZE)
    return args


def constructed_audit():
    # Choose the CURRENT auditor before the deepest completed.audit comparison.
    # Do not pass through the old466/465 wrappers that treat the tuple as a dict.
    return project_function(original_full.actual_audit,(
        ('from audit425 import audit','from audit470 import audit'),
    ))


def check_packet(packet,completed):
    if not isinstance(packet,tuple)or len(packet)!=3:
        raise RuntimeError('Original supervisor three-item audit ABI required')
    actual,kind,evidence=packet;wait=completed['qualification_rpc_wait']
    if not(wait['original_parent_timeout_seconds']==6000
           and wait['qualification_receive_timeout_seconds']==21600
           and wait['outer_supervisor_total_limit_seconds']==21600
           and wait['original_engine_configuration_object_unchanged']is True
           and wait['worker_configuration_changed']is False and wait['production_timeout_changed']is False):
        raise RuntimeError('Declared qualification-only wait and unchanged worker configuration required')
    if (actual!=completed['audit']or kind!='passed_current_canonical_full_AR'
            or evidence['whole_workload_AR_qualified']is not True
            or evidence['hidden_KV_recomputed_by_HOST']is not False
            or actual.get('ROOT470_original466_all_checks_retained')is not True
            or len(actual.get('ROOT470_actual_target_span_copy',[]))!=3):
        raise RuntimeError('Complete current numerical, AR and span-copy audit required')
    return packet


def actual_audit(directory):
    packet=constructed_audit()(directory)
    return check_packet(packet,json.loads((Path(directory)/'completed.json').read_text()))


def execution_namespace():
    ns=parent.execution_namespace()
    ns.update(prepared=prepared,storage=storage,launch_argv=launch_argv,actual_audit=actual_audit)
    from guard465 import parent as previous
    ns['execute_locked']=previous.replace_constants(ns['execute_locked'],(
        ('ROOT.V466.current-card-cohort48-full-own-AR-supervisor.v1',
         'ROOT.V470.current-card-span-copy-full-own-AR-supervisor.v1'),
    ),ns)
    ns['SCOPE']=dict(ns['SCOPE'],target_copy_programs=64,
        original_native_tree_update_method_unchanged=True,
        original_model_weights_admission_and_clocks_unchanged=True,
        new_target_copy_installed_before_all_original_node_checks=True,
        deepest_current_auditor_and_original_three_tuple_ABI=True)
    if ns['execute_locked'].__globals__ is not ns:raise RuntimeError('Actual owned namespace required')
    return ns


def main():
    global EXPECTED_FREEZE,EXPECTED_ARCHIVE
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--expected-freeze-sha256',required=True);parser.add_argument('--expected-archive-sha256',required=True)
    args=parser.parse_args();EXPECTED_FREEZE=args.expected_freeze_sha256;EXPECTED_ARCHIVE=args.expected_archive_sha256
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)or any(not re.fullmatch('[a-f0-9]{64}',v)for v in(EXPECTED_FREEZE,EXPECTED_ARCHIVE)):
        raise RuntimeError('Fresh safe tag and explicit source pins required')
    prepared();ns=execution_namespace()
    with LOCK.open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ns['execute_locked'](args.tag)


if __name__=='__main__':main()
