"""ROOT sole-owner full215; unchanged original verifier/limits/owned cleanup."""
import argparse,ast,copy,datetime,fcntl,gzip,hashlib,importlib.util,json,os,re,resource,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
from check215 import P as HERE,B,DONOR,require,ready,read,sha
from audit_reference215 import audit as reference_audit
if str(DONOR)not in sys.path:sys.path.insert(1,str(DONOR))
LOCK=B/'.ROOT_NPU_EXECUTION.lock';MAX_SECONDS=7200
MAX_TAG_BYTES=256*1024**2;MAX_AGGREGATE_BYTES=512*1024**2
EXPECTED_FREEZE=None;EXPECTED_ARCHIVE=None
ACTION='causal80-N9-to15-allB4-empty-active32-full463'
_DONOR_GUARD=None


def donor_guard():
    global _DONOR_GUARD
    if _DONOR_GUARD is None:
        p=B/'budget-diagnostics/v209-root-execution/guard209.py'
        f=read(DONOR/'FROZEN_SOURCE209.json')
        require(sha(p)==f['execution_guard_inputs'][str(p)],'exact canonical209 supervisor')
        spec=importlib.util.spec_from_file_location('_ROOT215_original_guard209',p)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);_DONOR_GUARD=m
    return _DONOR_GUARD


def prepared():
    f=ready(EXPECTED_FREEZE);sources=dict(f['files']);sources[str(HERE/'FREEZE215.json')]=EXPECTED_FREEZE
    archive=HERE/'SOURCE215.tar.gz'
    require(sha(archive)==EXPECTED_ARCHIVE,'explicit ROOT source archive SHA')
    with tarfile.open(archive,'r:gz')as saved:
        members=saved.getmembers()
        require(len(members)==len(sources)and {m.name for m in members}=={p.lstrip('/')for p in sources},'exact regular source archive set')
        for m in members:require(m.isfile()and hashlib.sha256(saved.extractfile(m).read()).hexdigest()==sources['/'+m.name],'archived input '+m.name)
    _,old_sources,dependency=donor_guard().prepared()
    require(all(f['files'].get(p)==h for p,h in old_sources.items()),'all canonical209 frozen source/evidence retained')
    return f,sources,dict(own_freeze_sha256=EXPECTED_FREEZE,own_archive_sha256=EXPECTED_ARCHIVE,
        original209_dependency=dependency,source_peer_review=False,canonical209_source_unchanged=True,
        device_qualification_inherited=False,production_delta='same213 plus budget-cost repair: mixed16 B3/B4 to allB4 only within original limits; clients/model/clock unchanged')


def storage(tag=None):
    old=donor_guard();actual=0 if tag is None else old.bytes_under(DONOR/'actual'/tag)
    total=old.bytes_under(DONOR)+old.bytes_under(HERE)
    require(actual<=MAX_TAG_BYTES and total<=MAX_AGGREGATE_BYTES,'private256MiB/512MiB bounds')
    return dict(actual_tag_bytes=actual,aggregate_private_bytes=total)


def launch_argv(tag):
    return ['/usr/bin/env','-i','HOME=/root','USER=root','LOGNAME=root','SHELL=/bin/bash','LANG=C.UTF-8',
        'PATH=/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin',
        'SPECSLO_V215_ROOT_FREEZE_SHA='+EXPECTED_FREEZE,'/bin/bash',str(HERE/'launch215.sh'),'--action',ACTION,'--tag',tag]


def actual_audit(directory):
    actual,status,arithmetic=donor_guard().actual_audit(directory)
    snapshots=[]
    for rank in range(4):
        r=read(Path(directory)/f'rank{rank}-local-diagnostic.json')
        cache=r['ROOT215_cached_parent_dispatch']
        require(r['overlay']['ROOT215_overlay']=='cached-parent-SLO-admission1-cost-aware-mixed16' and r['overlay']['SLO_AWARE_ADMISSION']is True
            and cache['initial_parent_source_checks']==1 and cache['cached_parent_returns']>1
            and cache['canonical_mapper_or_base_module_mutated']is False,'actual fresh215 overlay/cache used')
        require(all(r[name]['ROOT215_explicit_overlay_sources'] for name in ('actual_helper_origins_before_constructor','actual_helper_origins_after_constructor')),'explicit new source provenance on all4')
        trace=json.loads(gzip.decompress((Path(directory)/'windows/window32/full463'/f'rank{rank}-full463-actual-trace.json.gz').read_bytes()))
        require(trace['decode_counters']['spec_rhythm_slo_aware_admission_enabled']==1
            and trace['decode_counters']['spec_rhythm_target_slo_first_enabled']==0,
            'actual all4 full463 Native SLO-aware admission1 and targetSLO-first0')
        from cost215 import audit_snapshot
        audit_snapshot(cache, trace['source_generation'])
        require(r['overlay']['cost_aware_mixed16_budget']is True,'actual cost-aware overlay')
        reference=reference_audit(directory) if rank==0 else reference
        require(r['postfull_independent_FIA_AR']['completed']is True
            and r['postfull_independent_FIA_AR']==read(Path(directory)/'postfull-independent-FIA-AR'/f'rank{rank}-summary.json'),'complete exact postfull own-FIA reference')
        snapshots.append(cache)
    return actual,status,dict(arithmetic,ROOT215_cached_dispatch_snapshots=snapshots,
        changed_model_or_arrival_or_clock=False,budget_grants_may_change=True,SLO_AWARE_ADMISSION=True,postfull_independent_FIA_AR=reference)


def execution_namespace():
    old=donor_guard();namespace=dict(vars(old),P=DONOR,G=HERE,ACTION=ACTION,prepared=prepared,
        storage=storage,launch_argv=launch_argv,actual_audit=actual_audit,
        MAX_SECONDS=MAX_SECONDS,MAX_TAG_BYTES=MAX_TAG_BYTES,MAX_AGGREGATE_BYTES=MAX_AGGREGATE_BYTES,
        SCOPE=dict(cache_and_original_SLO_admission1_overlay215=True,source_donor209=True,full463=True,
            original_budget_limits_and_arrivals=True,cost_aware_mixed16_budget=True,postfull_independent_FIA_AR=True,performance_clock_ends_before_reference=True,active_window=32,new_target_keys=0,
            fixed_gamma_control_rerun=False,whole_workload_AR_qualified=False,formal_Goodput=False,adopted=False))
    text=(B/'budget-diagnostics/v154-root-execution/supervise_v154.py').read_text()
    changes=(('V154.root-owned-local-component-supervisor.v1','ROOT.V215.cached-parent-full-supervisor.v1'),
        ('V154 supervisor received','ROOT215 supervisor received'),('numeric_source_obligations_only=True','numeric_source_obligations_only=False'),
        ('one natural-positive R1 discovery, conditional new-union/child/next; no two complete zeros or full qualification',
         'same209 full463 after fresh original201/44/114+B4/M80/multiowner/7N/warm/2zero; cached mapper plus original SLO-aware-admission1 and cost-aware mixed16 budget'),
        ('bounded1800second','bounded7200second'),('within1800seconds','within7200seconds'),
        ("'actual_component_audit'","'actual_full_audit'"),("'actual_local_coverage_status'","'actual_full_status'"))
    for before,after in changes:
        require(text.count(before)==1,'exact supervisor metadata anchor '+before);text=text.replace(before,after)
    names={'verifier_report','child_limits','execute_locked'}
    nodes=[copy.deepcopy(n)for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name in names]
    require({n.name for n in nodes}==names,'original verifier/limits/supervisor bodies')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(HERE/'guard215.py')+'::original-supervision','exec'),namespace)
    return namespace


def main():
    global EXPECTED_FREEZE,EXPECTED_ARCHIVE
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--expected-freeze-sha256',required=True);parser.add_argument('--expected-archive-sha256',required=True)
    a=parser.parse_args();EXPECTED_FREEZE=a.expected_freeze_sha256;EXPECTED_ARCHIVE=a.expected_archive_sha256
    require(all(re.fullmatch('[0-9a-f]{64}',s)is not None for s in(EXPECTED_FREEZE,EXPECTED_ARCHIVE))and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',a.tag)is not None,'explicit source/archive pins and fresh safe tag')
    prepared();ns=execution_namespace();lock=LOCK.open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ns['execute_locked'](a.tag)
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
