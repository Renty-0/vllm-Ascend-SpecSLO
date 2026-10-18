"""One source-bound baseline cell of the predeclared comparison, ROOT-owned.

Importing this module performs no model, subprocess or device action. Tree
cells use their own complete guard and are deliberately excluded here.
"""
import argparse,datetime,fcntl,hashlib,importlib.util,json,re,signal,subprocess,time,traceback
from pathlib import Path
from final_common_comparison import ORDERS,collect,need
from prepare_final_comparison_plan import FIXED,OFFICIAL,verify_prepared
from final_owned_processes import Owned,snapshot
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
HELPER=Path('/root/SpecSLO/resume-20261012/run_baseline_calibration-v1.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_plan(path,expected,block,position):
    need(sha(path)==expected,'explicit immutable final command plan')
    plan=json.loads(Path(path).read_text());need(len(plan['cells'])==9,'all nine predeclared cells in plan')
    need(1<=block<=3 and 0<=position<=2,'one exact predeclared cell coordinate')
    cell=plan['cells'][(block-1)*3+position]
    need(cell['block']==block and cell['position']==position
         and cell['method']==ORDERS[block-1][position]
         and cell['method']in('fixed_gamma4','official_MC2_ON'),'baseline only; no outer tree flock')
    need(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*',cell['tag'])is not None,'safe fresh baseline tag')
    selected_path=Path(plan['selection_path']);need(sha(selected_path)==plan['selection_sha256'],'unchanged qualified candidate selection')
    selected=json.loads(selected_path.read_text())
    for name,wanted in selected['source_and_evidence_hashes'].items():need(sha(name)==wanted,'selected source/evidence retained')
    if cell['method']=='fixed_gamma4':
        expected_argv=['bash',str(FIXED/'run.sh'),str(Path(path).parent/'fixed-results'),cell['tag']]
        binding=verify_prepared(FIXED,FIXED/'PREPARED_SOURCE_MANIFEST.json',plan['fixed_prepared_binding']['manifest_sha256'])
        need(binding==plan['fixed_prepared_binding'],'same frozen fixed-gamma4 baseline')
    else:
        expected_argv=['bash',str(OFFICIAL/'run.sh'),'1',cell['tag']]
        binding=verify_prepared(OFFICIAL,OFFICIAL/'PREPARED_PUBLICATION_V2_SOURCE_MANIFEST.json',plan['official_prepared_binding']['manifest_sha256'])
        need(binding==plan['official_prepared_binding'],'same actual MC2 official baseline')
    need(cell['argv']==expected_argv,'only the fixed reviewed baseline entry and configuration')
    return plan,cell


def native_verifier(directory,phase):
    argv=['bash',str(B/'serial-v1/runtime.sh'),'python','-B',
          '/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/verify_lock.py',
          '/root/data/specslo-baselines/nano-pearl-ascend-v1/lock.json']
    result=subprocess.run(argv,capture_output=True,text=True,timeout=180)
    path=directory/(phase+'-native-source-verifier.log')
    with path.open('x')as stream:stream.write(result.stdout+result.stderr)
    values=[]
    for line in result.stdout.splitlines():
        try:value=json.loads(line)
        except json.JSONDecodeError:continue
        if type(value)is dict and 'checked_frozen_files'in value:values.append(value)
    need(result.returncode==0 and len(values)==1 and values[0]['status']=='pass'
         and values[0]['errors']==[]and values[0]['checked_frozen_files']==10496
         and values[0]['checked_external_files']==15 and values[0]['checked_packages']==212,
         'unchanged original native source/environment10496/15/212 verification')
    return dict(argv=argv,exit_code=result.returncode,path=str(path),sha256=sha(path),actual_result=values[0])


def result_and_metrics(cell):
    paths=cell['outputs'];result=Path(paths['result']);need(result.is_file(),'actual complete baseline result')
    if cell['method']=='fixed_gamma4':
        gate=json.loads(Path(paths['mechanism_gate']).read_text())
        need(gate['status']=='pass'and gate['errors']==[]and gate['result_sha256']==sha(result),
             'unchanged original fixed production mechanism gate and exact result bytes')
        metrics=collect('fixed_gamma4',result,witness_directory=paths['witness_directory'])
        return metrics,dict(original_mechanism_gate=gate)
    receipt=json.loads(Path(paths['receipt']).read_text());audit=json.loads(Path(paths['original_audit']).read_text())
    need(receipt['status']=='complete'and receipt['original_auditor_returncode']==0
         and receipt['cleanup']['after']==[]and receipt['source_before']['passed']is True
         and receipt['source_after']['passed']is True,'actual original official wrapper and unchanged auditor passed')
    need(audit['passed']is True and audit['result_sha256']==sha(result)and audit['MC2_enabled']is True
         and audit['selected_MC2_modules_per_rank']==[128]*4,'all four actual MC2 projection-module selections')
    witnesses=[]
    for path in Path(paths['witness_directory']).glob('generation-*.json.gz'):
        import gzip
        value=json.loads(gzip.decompress(path.read_bytes()))
        if len(value.get('source_rows',[]))==463:witnesses.append(path)
    need(len(witnesses)==1,'one complete actual engine-publication witness')
    metrics=collect('official_MC2_ON',witnesses[0])
    return metrics,dict(original_official_audit=audit,ordinary_MC2_actually_enabled=True,
                        MC2_projection_modules_by_rank=[128]*4,official_receipt_sha256=sha(paths['receipt']))


def execute(plan_path,expected,block,position,source_manifest,source_sha):
    need(sha(source_manifest)==source_sha,'explicit frozen ROOT comparison-tool manifest')
    source_pins=json.loads(Path(source_manifest).read_text())['files']
    need(str(Path(__file__).resolve())in source_pins and str(HELPER)in source_pins,'own root guard and original owned helper pinned')
    def sources():
        for name,wanted in source_pins.items():need(sha(name)==wanted,'unchanged ROOT comparison source '+name)
    sources();plan,cell=checked_plan(plan_path,expected,block,position)
    root=Path(plan_path).resolve().parent/'baseline-controls'/cell['tag']
    need(root.is_relative_to(B)and not root.exists(),'fresh private baseline control output')
    lock=(B/'.ROOT_NPU_EXECUTION.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root.mkdir(parents=True,mode=0o700)
    row=dict(schema='SpecSLO.final-single-baseline-cell.v1',status='running',block=block,position=position,
        method=cell['method'],tag=cell['tag'],plan_path=str(plan_path),plan_sha256=expected,
        source_manifest=str(source_manifest),source_manifest_sha256=source_sha,
        created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),errors=[],physical_cards=[2,5,6,7])
    def save():
        tmp=root/'receipt.tmp';tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(root/'receipt.json')
    def interrupted(sig,frame):raise RuntimeError('ROOT baseline cell interrupted '+str(sig))
    handlers={sig:signal.signal(sig,interrupted)for sig in(signal.SIGTERM,signal.SIGINT)}
    child=None;owned=None;h=None;save()
    try:
        spec=importlib.util.spec_from_file_location('_final_baseline_owned_helper',HELPER)
        h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);h.PHYSICAL=(2,5,6,7)
        row['devices_before']=h.device_state();need(row['devices_before']['all_idle']is True,'same four physical cards actually idle')
        row['pre_verifier']=native_verifier(root,'pre');sources()
        row['argv']=cell['argv']
        with(root/'child.log').open('x')as log:
            child=subprocess.Popen(cell['argv'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            owned=Owned(child.pid,snapshot());row['child_pid']=child.pid;start=time.monotonic();save()
            while child.poll()is None:
                owned.update(snapshot());row['elapsed_seconds']=time.monotonic()-start
                need(row['elapsed_seconds']<2700,'bounded baseline cell; inner original limits stay unchanged')
                time.sleep(1)
            row['child_exit_code']=child.wait()
        need(row['child_exit_code']==0,'original baseline entry completed')
        row['metrics'],row['original_result_audit']=result_and_metrics(cell)
        row['actual_result_complete']=True
    except BaseException:row['errors'].append(traceback.format_exc())
    finally:
        if owned is not None:
            row['owned_cleanup']=owned.cleanup();row['owned_remaining']=row['owned_cleanup']['after']
        else:row['owned_remaining']=[]
        if child is not None:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:row['errors'].append('owned baseline child could not be reaped')
        try:
            sources();checked_plan(plan_path,expected,block,position);row['sources_unchanged']=True
            row['post_verifier']=native_verifier(root,'post')
            row['devices_after']=h.device_state()if h is not None else None
            need(not row['owned_remaining']and row['devices_after']is not None
                 and row['devices_after']['all_idle']is True,'all owned workers exited and actual four cards idle')
        except BaseException:row['errors'].append(traceback.format_exc())
        row['status']='complete'if not row['errors']and row.get('actual_result_complete')else'failed'
        row['finished_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
        for sig,handler in handlers.items():signal.signal(sig,handler)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    print(json.dumps(dict(status=row['status'],receipt=str(root/'receipt.json'),errors=row['errors'])))
    return 0 if row['status']=='complete'else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--expected-plan-sha256',required=True);parser.add_argument('--block',type=int,required=True)
    parser.add_argument('--position',type=int,required=True);parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--expected-source-manifest-sha256',required=True);a=parser.parse_args()
    raise SystemExit(execute(a.plan,a.expected_plan_sha256,a.block,a.position,a.source_manifest,a.expected_source_manifest_sha256))
