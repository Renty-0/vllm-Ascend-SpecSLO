"""ROOT bounded same-prefix neural diagnostic, original owned cleanup."""
import argparse,ast,copy,datetime,fcntl,gzip,hashlib,importlib.util,json,os,re,resource,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
from check317 import P as HERE,B,DONOR,OLD,OLD_SHA,require,ready,read,sha
from guard218 import donor_guard
LOCK=B/'.ROOT_NPU_EXECUTION.lock';MAX_SECONDS=2700
MAX_TAG_BYTES=256*1024**2;MAX_AGGREGATE_BYTES=2048*1024**2
EXPECTED_FREEZE=None;EXPECTED_ARCHIVE=None;ACTION='alternate-chain-B1to8-M165-N32-node-component'

def prepared():
    original=ready('36cbac2bca38b3b69b2ab4ed04f984bb48782fd51d5bb682d04f3d98ba76405f')
    require(sha(HERE/'FREEZE317_ABI311.json')==EXPECTED_FREEZE,'explicit additive ABI311 source binding')
    f=read(HERE/'FREEZE317_ABI311.json')
    for path,wanted in f['files'].items():require(sha(path)==wanted,'ABI311 source unchanged '+path)
    require(all(f['files'].get(path)==wanted for path,wanted in original['files'].items()),'entire original317 numeric body retained')
    sources=dict(f['files']);sources[str(HERE/'FREEZE317_ABI311.json')]=EXPECTED_FREEZE
    archive=HERE/'SOURCE317_ABI311.tar.gz'
    require(sha(archive)==EXPECTED_ARCHIVE,'explicit255 source archive SHA')
    with tarfile.open(archive,'r:gz')as saved:
        members=saved.getmembers()
        require(len(members)==len(sources)and {m.name for m in members}=={p.lstrip('/')for p in sources},'exact255 regular source archive set')
        for m in members:require(m.isfile()and hashlib.sha256(saved.extractfile(m).read()).hexdigest()==sources['/'+m.name],'archived255 input '+m.name)
    original=read(OLD/'FREEZE218.json')
    require(all(f['files'].get(p)==h for p,h in original['files'].items()),'entire original218 source and evidence retained')
    return f,sources,dict(own_freeze_sha256=EXPECTED_FREEZE,own_archive_sha256=EXPECTED_ARCHIVE,original218_freeze_sha256=OLD_SHA,
        device_qualification_inherited=False,purpose='independently prefilled target-only fixed M165 independent node math and paired component cost; no workload AR or performance qualification')

def storage(tag=None):
    old=donor_guard();actual=0 if tag is None else old.bytes_under(DONOR/'actual'/tag)
    total=old.bytes_under(DONOR)+old.bytes_under(HERE)
    require(actual<=MAX_TAG_BYTES and total<=MAX_AGGREGATE_BYTES,'private256MiB/2048MiB diagnostic limits')
    return dict(actual_tag_bytes=actual,aggregate_private_bytes=total)

def launch_argv(tag):
    return ['/usr/bin/env','-i','HOME=/root','USER=root','LOGNAME=root','SHELL=/bin/bash','LANG=C.UTF-8',
        'PATH=/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin',
        'SPECSLO_V317_ROOT_FREEZE_SHA=36cbac2bca38b3b69b2ab4ed04f984bb48782fd51d5bb682d04f3d98ba76405f','/bin/bash',str(HERE/'launch317_abi311.sh'),'--action',ACTION,'--tag',tag]

def actual_audit(directory):
    directory=Path(directory)
    reports=[read(directory/f'rank{r}-numeric.json')for r in range(4)]
    installed=[read(directory/f'rank{r}-installed-dispatch-before-model.json')for r in range(4)]
    require([v['rank']for v in installed]==list(range(4))and all(v['actual_code_reconstructed_equal']is True and v['models_loaded']is False for v in installed)
        and len({v['source_SHA']for v in installed})==1,'all4 actual replay-first bodies verified BEFORE models loaded')
    require(read(directory/'worker-exit.json')==dict(closed=True,all_exited=True),'all diagnostic roles exited')
    require(read(directory/'completed.json')==dict(replies=[['calibrated',r,dict(numeric_cases=546,performance=False)]for r in range(4)],performance=False,adopted=False),'complete original diagnostic protocol;546 actual model cases')
    require([v['rank']for v in reports]==list(range(4)),'all four diagnostic reports')
    require(all(v['model_dtype']=='torch.bfloat16'for v in reports),'all four actual model BF16 dtype retained')
    require(all(v['performance']is False and v['whole_workload_AR_qualified']is False and v['adopted']is False for v in reports),'no production or performance qualification')
    require(all(len(v['math']['cases'])==546 and v['math']['all_independent_prefixes_unchanged']is True and all(c['hidden_bitwise_equal']and c['KV_bitwise_equal']and c['exact_greedy']and c['graph_eager_KV_bitwise_equal']for c in v['math']['cases'])for v in reports[1:]),'all546 actual model cases hidden/greedy/all128 KV bitwise equal, all96 prefixes unchanged')
    require(all(len(v['fresh165_raw_operator']['checks'])==144 and v['fresh165_raw_operator']['raw16_bit_coverage']=={'256':65536,'384':65536} for v in reports[1:]),'all3 target ranks freshly prove165 raw-cache programs, both widths, independent strided inputs and slot dtypes')
    expected=read(HERE/'DECLARED317_CASES.json')['cases']
    require(len(expected)==546,'predeclared actual388 node cases')
    for report in reports[1:]:
        require([r['case']for r in report['math']['cases']]==expected,'all declared old227 plus161 alternate cases executed')
        from audit_static_calls317 import audit_calls
        require(report['dispatch_audit']==audit_calls(expected,report['actual_static_graph_calls']),
                'actual original capture/change/restore and real call reconstruction')
        require(len(report['paired_target_cost']['rows'])==16,'all declared synchronous target cost configurations')
        require(report['all_prefixes_still_exact_after_costs']is True,'all96 actual prefixes protected after timing as well')
        for row in report['paired_target_cost']['rows']:
            require(row['rounds']==20 and len(row['pairs'])==20 and row['original_greedy_output_stable'],
                    'all20 rotated real model cost pairs and stable greedy outputs')
    return dict(status='bounded_M165_all546_node_cases_completed',reports=[dict(rank=v['rank'],numeric_cases=len(v['math']['cases']),path=str(directory/f"rank{v['rank']}-numeric.json"),sha256=sha(directory/f"rank{v['rank']}-numeric.json"))for v in reports],formal_Goodput=False,adopted=False),'diagnostic_only',dict(original_startup_inventory_not_run=True,actual_model_cases546=True,six_grid_families=True)


def execution_namespace():
    old=donor_guard();namespace=dict(vars(old),P=DONOR,G=HERE,ACTION=ACTION,prepared=prepared,storage=storage,
        launch_argv=launch_argv,actual_audit=actual_audit,MAX_SECONDS=MAX_SECONDS,
        MAX_TAG_BYTES=MAX_TAG_BYTES,MAX_AGGREGATE_BYTES=MAX_AGGREGATE_BYTES,
        SCOPE=dict(shared_prefix_numeric_diagnostic=True,performance=False,whole_workload_AR_qualified=False,formal_Goodput=False,adopted=False))
    source=(B/'budget-diagnostics/v154-root-execution/supervise_v154.py').read_text()
    for before,after in (
        ('V154.root-owned-local-component-supervisor.v1','ROOT.V317.bounded-numeric-probe-supervisor.v1'),
        ('V154 supervisor received','ROOT317 supervisor received'),
        ('one natural-positive R1 discovery, conditional new-union/child/next; no two complete zeros or full qualification',
         'actual original cohort1 independent real prefixes; grouped and per-query target math comparison; no performance timing; diagnostic only')):
        require(source.count(before)==1,'supervisor metadata anchor '+before);source=source.replace(before,after)
    names={'verifier_report','child_limits','execute_locked'}
    nodes=[copy.deepcopy(n)for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name in names]
    require({n.name for n in nodes}==names,'original actual verifier, limits and cleanup supervisor')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(HERE/'guard317_abi311.py')+'::original-supervision','exec'),namespace)
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
