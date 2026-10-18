"""ROOT bounded same-prefix neural diagnostic, original owned cleanup."""
import argparse,ast,copy,datetime,fcntl,gzip,hashlib,importlib.util,json,os,re,resource,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
from check228 import P as HERE,B,DONOR,OLD,OLD_SHA,require,ready,read,sha
from guard218 import donor_guard
LOCK=B/'.ROOT_NPU_EXECUTION.lock';MAX_SECONDS=1800
MAX_TAG_BYTES=256*1024**2;MAX_AGGREGATE_BYTES=512*1024**2
EXPECTED_FREEZE=None;EXPECTED_ARCHIVE=None;ACTION='canonical-FULL-numeric-body-cost'

def prepared():
    f=ready(EXPECTED_FREEZE);sources=dict(f['files']);sources[str(HERE/'FREEZE228.json')]=EXPECTED_FREEZE
    archive=HERE/'SOURCE228.tar.gz'
    require(sha(archive)==EXPECTED_ARCHIVE,'explicit228 source archive SHA')
    with tarfile.open(archive,'r:gz')as saved:
        members=saved.getmembers()
        require(len(members)==len(sources)and {m.name for m in members}=={p.lstrip('/')for p in sources},'exact228 regular source archive set')
        for m in members:require(m.isfile()and hashlib.sha256(saved.extractfile(m).read()).hexdigest()==sources['/'+m.name],'archived228 input '+m.name)
    original=read(OLD/'FREEZE218.json')
    require(all(f['files'].get(p)==h for p,h in original['files'].items()),'entire original218 source and evidence retained')
    return f,sources,dict(own_freeze_sha256=EXPECTED_FREEZE,own_archive_sha256=EXPECTED_ARCHIVE,original218_freeze_sha256=OLD_SHA,
        device_qualification_inherited=False,purpose='original cohort incremental common history numeric forensics; no performance or independent AR qualification')

def storage(tag=None):
    old=donor_guard();actual=0 if tag is None else old.bytes_under(DONOR/'actual'/tag)
    total=old.bytes_under(DONOR)+old.bytes_under(HERE)
    require(actual<=MAX_TAG_BYTES and total<=MAX_AGGREGATE_BYTES,'private256MiB/512MiB diagnostic limits')
    return dict(actual_tag_bytes=actual,aggregate_private_bytes=total)

def launch_argv(tag):
    return ['/usr/bin/env','-i','HOME=/root','USER=root','LOGNAME=root','SHELL=/bin/bash','LANG=C.UTF-8',
        'PATH=/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin',
        'SPECSLO_V228_ROOT_FREEZE_SHA='+EXPECTED_FREEZE,'/bin/bash',str(HERE/'launch228.sh'),'--action',ACTION,'--tag',tag]

def actual_audit(directory):
    directory=Path(directory);reports=[read(directory/f'rank{r}-numeric.json')for r in range(4)]
    require(read(directory/'worker-exit.json')==dict(closed=True,all_exited=True),'all canonical attention diagnostic roles exited')
    require(read(directory/'completed.json')==dict(replies=[['calibrated',r,dict(numeric_cases=4,performance=False)]for r in range(4)],performance=False,adopted=False),'all4 canonical3-context workers')
    tokens=[]
    for rank,report in enumerate(reports):
        require(report['rank']==rank and len(report['cases'])==4 and report['observer_installed']is False
            and report['decode_attention_metadata_changed']is True and report['original_greedy_sampler_unchanged']is True,'explicit diagnostic-only decode metadata variation')
        if rank:
            fresh=[r for c in report['cases']if c['kind']=='fresh'for r in c['detail']['frames']];hist=[c['detail']['routes']for c in report['cases']if c['kind']=='history']
            require(all(r['shape']['tree']is True and r['shape']['M']==80 and r['shape']['S']==80 and r['shape']['q']==list(range(1,81)) for r in fresh),'actual forced FULL path for originally branched and linear inputs')
            require(all(r['future_canary_query_hidden_and_common_KV_exact']is True for rows in hist for r in rows),'actual canonical histories future invariance')
            require(all(r['current_real16_exact_metadata_original_matcher_passed']is True for rows in hist for r in rows),'all80 real-row exact-length FULL original Native model matcher')
            cost=hist[-1][-1]['dynamic_attention_body_cost']
            require(cost['M']==80 and cost['real_owners']==16 and len(cost['samples'])==12
                and cost['alternating_two_adjacent_context_frames']is True,'actual dynamic metadata body ABBA timing')
            require(all(r['execution']['mode']=='replay'and r['host_wall_ms_per_call']>0 and r['device_ms_per_call']>0 for r in cost['samples']),'actual finite original graph replay timings')
            tokens.append([[r['diagnostic_refined_token']for r in rows]for rows in[fresh,*hist]])
    require(tokens[0]==tokens[1]==tokens[2],'all3 target heads agree; cross-history hypothesis may pass or fail')
    value=dict(status='passed_bounded_canonical_FULL_observations',ranks=4,cases=4,refined_tokens_by_case=tokens[0],
        all3_targets_agree=True,cross_history_equality_not_assumed=True,isolated_dynamic_body_cost_only=True,
        independent_AR_qualified=False,formal_Goodput=False,adopted=False)
    return value,'diagnostic_only',dict(neural_diagnostic_only=True,full_production_qualification=False)

def execution_namespace():
    old=donor_guard();namespace=dict(vars(old),P=DONOR,G=HERE,ACTION=ACTION,prepared=prepared,storage=storage,
        launch_argv=launch_argv,actual_audit=actual_audit,MAX_SECONDS=MAX_SECONDS,
        MAX_TAG_BYTES=MAX_TAG_BYTES,MAX_AGGREGATE_BYTES=MAX_AGGREGATE_BYTES,
        SCOPE=dict(shared_prefix_numeric_diagnostic=True,performance=False,whole_workload_AR_qualified=False,formal_Goodput=False,adopted=False))
    source=(B/'budget-diagnostics/v154-root-execution/supervise_v154.py').read_text()
    for before,after in (
        ('V154.root-owned-local-component-supervisor.v1','ROOT.V228.bounded-numeric-probe-supervisor.v1'),
        ('V154 supervisor received','ROOT228 supervisor received'),
        ('one natural-positive R1 discovery, conditional new-union/child/next; no two complete zeros or full qualification',
         'three prior contexts with forced FULL9 decode metadata and uninstalled FP32 tie observer; dynamic body replay ABBA cost; diagnostic only')):
        require(source.count(before)==1,'supervisor metadata anchor '+before);source=source.replace(before,after)
    names={'verifier_report','child_limits','execute_locked'}
    nodes=[copy.deepcopy(n)for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name in names]
    require({n.name for n in nodes}==names,'original actual verifier, limits and cleanup supervisor')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(HERE/'guard228.py')+'::original-supervision','exec'),namespace)
    return namespace

def main():
    global EXPECTED_FREEZE,EXPECTED_ARCHIVE
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--expected-freeze-sha256',required=True);parser.add_argument('--expected-archive-sha256',required=True)
    args=parser.parse_args();EXPECTED_FREEZE=args.expected_freeze_sha256;EXPECTED_ARCHIVE=args.expected_archive_sha256
    require(all(re.fullmatch('[0-9a-f]{64}',s)is not None for s in(EXPECTED_FREEZE,EXPECTED_ARCHIVE))
        and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',args.tag)is not None,'explicit pins and safe fresh diagnostic tag')
    prepared();namespace=execution_namespace();lock=LOCK.open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);namespace['execute_locked'](args.tag)
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
