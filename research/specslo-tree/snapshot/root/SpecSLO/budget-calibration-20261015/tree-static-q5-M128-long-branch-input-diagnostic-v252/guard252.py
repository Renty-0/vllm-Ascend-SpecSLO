"""ROOT bounded same-prefix neural diagnostic, original owned cleanup."""
import argparse,ast,copy,datetime,fcntl,gzip,hashlib,importlib.util,json,os,re,resource,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
from check252 import P as HERE,B,DONOR,OLD,OLD_SHA,require,ready,read,sha
from guard218 import donor_guard
LOCK=B/'.ROOT_NPU_EXECUTION.lock';MAX_SECONDS=1800
MAX_TAG_BYTES=256*1024**2;MAX_AGGREGATE_BYTES=768*1024**2
EXPECTED_FREEZE=None;EXPECTED_ARCHIVE=None;ACTION='static-q5-M128-real-prefix-component'

def prepared():
    f=ready(EXPECTED_FREEZE);sources=dict(f['files']);sources[str(HERE/'FREEZE252.json')]=EXPECTED_FREEZE
    archive=HERE/'SOURCE252.tar.gz'
    require(sha(archive)==EXPECTED_ARCHIVE,'explicit252 source archive SHA')
    with tarfile.open(archive,'r:gz')as saved:
        members=saved.getmembers()
        require(len(members)==len(sources)and {m.name for m in members}=={p.lstrip('/')for p in sources},'exact252 regular source archive set')
        for m in members:require(m.isfile()and hashlib.sha256(saved.extractfile(m).read()).hexdigest()==sources['/'+m.name],'archived252 input '+m.name)
    original=read(OLD/'FREEZE218.json')
    require(all(f['files'].get(p)==h for p,h in original['files'].items()),'entire original218 source and evidence retained')
    return f,sources,dict(own_freeze_sha256=EXPECTED_FREEZE,own_archive_sha256=EXPECTED_ARCHIVE,original218_freeze_sha256=OLD_SHA,
        device_qualification_inherited=False,purpose='independently prefilled target-only fixed M128 independent node math and paired component cost; no workload AR or performance qualification')

def storage(tag=None):
    old=donor_guard();actual=0 if tag is None else old.bytes_under(DONOR/'actual'/tag)
    total=old.bytes_under(DONOR)+old.bytes_under(HERE)
    require(actual<=MAX_TAG_BYTES and total<=MAX_AGGREGATE_BYTES,'private256MiB/512MiB diagnostic limits')
    return dict(actual_tag_bytes=actual,aggregate_private_bytes=total)

def launch_argv(tag):
    return ['/usr/bin/env','-i','HOME=/root','USER=root','LOGNAME=root','SHELL=/bin/bash','LANG=C.UTF-8',
        'PATH=/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin',
        'SPECSLO_V252_ROOT_FREEZE_SHA='+EXPECTED_FREEZE,'/bin/bash',str(HERE/'launch252.sh'),'--action',ACTION,'--tag',tag]

def actual_audit(directory):
    directory=Path(directory)
    reports=[read(directory/f'rank{r}-numeric.json')for r in range(4)]
    require(read(directory/'worker-exit.json')==dict(closed=True,all_exited=True),'all diagnostic roles exited')
    require(read(directory/'completed.json')==dict(replies=[['calibrated',r,dict(numeric_cases=120,performance=False)]for r in range(4)],performance=False,adopted=False),'complete original diagnostic protocol;120 is preserved protocol label only')
    require([v['rank']for v in reports]==list(range(4)),'all four diagnostic reports')
    require(all(v['performance']is False and v['whole_workload_AR_qualified']is False and v['adopted']is False for v in reports),'no production or performance qualification')
    require(all(len(v['cases'])==3 and v['input_trace']for v in reports[1:]),'three actual math cases and captured model attention inputs')
    return dict(status='bounded_long_branch_diagnostic_completed',reports=reports,formal_Goodput=False,adopted=False),'diagnostic_only',dict(original_startup_inventory_not_run=True,protocol_120_label_is_not_actual_node_count=True)


def execution_namespace():
    old=donor_guard();namespace=dict(vars(old),P=DONOR,G=HERE,ACTION=ACTION,prepared=prepared,storage=storage,
        launch_argv=launch_argv,actual_audit=actual_audit,MAX_SECONDS=MAX_SECONDS,
        MAX_TAG_BYTES=MAX_TAG_BYTES,MAX_AGGREGATE_BYTES=MAX_AGGREGATE_BYTES,
        SCOPE=dict(shared_prefix_numeric_diagnostic=True,performance=False,whole_workload_AR_qualified=False,formal_Goodput=False,adopted=False))
    source=(B/'budget-diagnostics/v154-root-execution/supervise_v154.py').read_text()
    for before,after in (
        ('V154.root-owned-local-component-supervisor.v1','ROOT.V252.bounded-numeric-probe-supervisor.v1'),
        ('V154 supervisor received','ROOT252 supervisor received'),
        ('one natural-positive R1 discovery, conditional new-union/child/next; no two complete zeros or full qualification',
         'actual original cohort1 independent real prefixes; grouped and per-query target math comparison; no performance timing; diagnostic only')):
        require(source.count(before)==1,'supervisor metadata anchor '+before);source=source.replace(before,after)
    names={'verifier_report','child_limits','execute_locked'}
    nodes=[copy.deepcopy(n)for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name in names]
    require({n.name for n in nodes}==names,'original actual verifier, limits and cleanup supervisor')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(HERE/'guard252.py')+'::original-supervision','exec'),namespace)
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
