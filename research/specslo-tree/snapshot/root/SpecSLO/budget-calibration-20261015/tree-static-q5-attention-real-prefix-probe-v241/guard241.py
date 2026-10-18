"""ROOT bounded same-prefix neural diagnostic, original owned cleanup."""
import argparse,ast,copy,datetime,fcntl,gzip,hashlib,importlib.util,json,os,re,resource,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
from check241 import P as HERE,B,DONOR,OLD,OLD_SHA,require,ready,read,sha
from guard218 import donor_guard
LOCK=B/'.ROOT_NPU_EXECUTION.lock';MAX_SECONDS=1800
MAX_TAG_BYTES=256*1024**2;MAX_AGGREGATE_BYTES=512*1024**2
EXPECTED_FREEZE=None;EXPECTED_ARCHIVE=None;ACTION='static-q5-attention-real-prefix'

def prepared():
    f=ready(EXPECTED_FREEZE);sources=dict(f['files']);sources[str(HERE/'FREEZE241.json')]=EXPECTED_FREEZE
    archive=HERE/'SOURCE241.tar.gz'
    require(sha(archive)==EXPECTED_ARCHIVE,'explicit241 source archive SHA')
    with tarfile.open(archive,'r:gz')as saved:
        members=saved.getmembers()
        require(len(members)==len(sources)and {m.name for m in members}=={p.lstrip('/')for p in sources},'exact241 regular source archive set')
        for m in members:require(m.isfile()and hashlib.sha256(saved.extractfile(m).read()).hexdigest()==sources['/'+m.name],'archived241 input '+m.name)
    original=read(OLD/'FREEZE218.json')
    require(all(f['files'].get(p)==h for p,h in original['files'].items()),'entire original218 source and evidence retained')
    return f,sources,dict(own_freeze_sha256=EXPECTED_FREEZE,own_archive_sha256=EXPECTED_ARCHIVE,original218_freeze_sha256=OLD_SHA,
        device_qualification_inherited=False,purpose='independently prefilled target-only canonical M80 node/branch math; no workload AR or performance qualification')

def storage(tag=None):
    old=donor_guard();actual=0 if tag is None else old.bytes_under(DONOR/'actual'/tag)
    total=old.bytes_under(DONOR)+old.bytes_under(HERE)
    require(actual<=MAX_TAG_BYTES and total<=MAX_AGGREGATE_BYTES,'private256MiB/512MiB diagnostic limits')
    return dict(actual_tag_bytes=actual,aggregate_private_bytes=total)

def launch_argv(tag):
    return ['/usr/bin/env','-i','HOME=/root','USER=root','LOGNAME=root','SHELL=/bin/bash','LANG=C.UTF-8',
        'PATH=/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin',
        'SPECSLO_V241_ROOT_FREEZE_SHA='+EXPECTED_FREEZE,'/bin/bash',str(HERE/'launch241.sh'),'--action',ACTION,'--tag',tag]

def actual_audit(directory):
    directory=Path(directory)
    reports=[read(directory/f'rank{r}-numeric.json')for r in range(4)]
    require(read(directory/'worker-exit.json')==dict(closed=True,all_exited=True),'all four diagnostic roles exited')
    require(read(directory/'completed.json')==dict(replies=[['calibrated',r,dict(numeric_cases=55,performance=False)]for r in range(4)],performance=False,adopted=False),'all diagnostic role replies')
    require([r['rank']for r in reports]==list(range(4)),'all role reports')
    require(all(r['whole_workload_AR_qualified']is False and r['performance']is False and r['adopted']is False for r in reports),'bounded diagnosis only')
    require(reports[0]['comparison']is None,'draft no target math')
    require(all(r['comparison']['original_observed_disagreement_reproduced'] is True for r in reports[1:]),'original real discrepancy reproduced on all target ranks')
    require(reports[1]['static']['heads']==reports[2]['static']['heads']==reports[3]['static']['heads'],'all three static greedy reports agree')
    return dict(status='bounded_real_prefix_diagnostic_completed',ranks=4,comparison=reports[1]['comparison'],static=reports[1]['static'],whole_workload_AR_qualified=False,formal_Goodput=False,adopted=False),'diagnostic_only',dict(independent_real_prefix_probe=True,original_startup201_44_114_not_run=True)


def execution_namespace():
    old=donor_guard();namespace=dict(vars(old),P=DONOR,G=HERE,ACTION=ACTION,prepared=prepared,storage=storage,
        launch_argv=launch_argv,actual_audit=actual_audit,MAX_SECONDS=MAX_SECONDS,
        MAX_TAG_BYTES=MAX_TAG_BYTES,MAX_AGGREGATE_BYTES=MAX_AGGREGATE_BYTES,
        SCOPE=dict(shared_prefix_numeric_diagnostic=True,performance=False,whole_workload_AR_qualified=False,formal_Goodput=False,adopted=False))
    source=(B/'budget-diagnostics/v154-root-execution/supervise_v154.py').read_text()
    for before,after in (
        ('V154.root-owned-local-component-supervisor.v1','ROOT.V241.bounded-numeric-probe-supervisor.v1'),
        ('V154 supervisor received','ROOT241 supervisor received'),
        ('one natural-positive R1 discovery, conditional new-union/child/next; no two complete zeros or full qualification',
         'actual original cohort1 independent real prefixes; grouped and per-query target math comparison; no performance timing; diagnostic only')):
        require(source.count(before)==1,'supervisor metadata anchor '+before);source=source.replace(before,after)
    names={'verifier_report','child_limits','execute_locked'}
    nodes=[copy.deepcopy(n)for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name in names]
    require({n.name for n in nodes}==names,'original actual verifier, limits and cleanup supervisor')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(HERE/'guard241.py')+'::original-supervision','exec'),namespace)
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
