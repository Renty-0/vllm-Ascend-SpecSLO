"""Terminal-only independent HOST/Torch-CPU audit; never starts a model/NPU."""
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
import tarfile

sys.dont_write_bytecode=True
B=Path('/root/SpecSLO/budget-calibration-20261015');P=B/'tree-upstream-control-v105';D=B/'budget-diagnostics'
ACTION='upstream-KV-attention-control'
FREEZE_SHA='ca31ff70729becfd5adb3075cd67edd92ffe344dbaab0312a779b93c2404b576'
CPU_SHA='5654fcd1a8af28974f72cb61768b04cc26e8163859c7522b3eb784e356a819f4'
PRE=D/'ROOT_V105_PRE_NPU_SOURCE_AND_HOST_AUDIT.json'
PRE_SHA='ce1b725acdc996b6e17694d1dcb3fca778f72a41775e67a1523564671c5c70ef'


def need(ok,label):
    if not ok:raise RuntimeError('V105 external actual audit refused: '+label)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def terminal(r,tag):
    need(r.get('status')=='complete' and r.get('errors')==[] and r.get('schema')=='V105.actual-upstream-supervisor.v1'
        and r.get('action')==ACTION and r.get('local_execution_complete') is True,'complete actual control only')
    need(r['child']['argv']==['bash',str(P/'launch_upstream_v105.sh'),'--action',ACTION,'--tag',tag]
        and r['child']['exit_code']==0 and r['owned_remaining']==[],'child command/reap/owned cleanup')
    need(r['formal_Goodput'] is r['online_qualified'] is False and r['physical_cards']==[2,5,6,7], 'diagnostic scope/cards')
    need(r['experimental_sources_unchanged'] is r['external_sources_unchanged'] is True,'actual final source checks')
    need(r['limits']==dict(seconds=3600,private_bytes=32*1024**2,file_bytes=16*1024**2)
        and 0<r['elapsed_seconds']<3600 and 0<r['private_bytes']<32*1024**2,'actual resource limits')
    for label in ('devices_before','devices_after'):
        row=r[label];hardware=row['hardware'];processes=row['processes']
        parsed=[(int(i),name) for i,name in re.findall(r'^\|\s*(\d+)\s+(\S+)\s*\|',hardware['output'],re.MULTILINE) if not name.isdecimal()]
        need(row['all_idle'] is row['four_910B2'] is True and sorted(parsed)==[(i,'910B2') for i in (2,5,6,7)]
            and hardware['exit_code']==0 and hardware['timed_out'] is False and set(processes)=={'2','5','6','7'},'actual four910B2 '+label)
        need(all(v['idle'] is True and v['exit_code']==0 and v['timed_out'] is False
            and re.search(r'No process in device\.|No running processes found',v['output'],re.I) for v in processes.values()),'actual idle per-card raw process evidence')
    for label in ('pre_verifier','post_verifier'):
        row=r[label];need(row['exit_code']==0 and row['log_sha256']==sha(row['log'])
            and row['log_bytes']==Path(row['log']).stat().st_size,'actual original lock '+label)


def archive(path,expected):
    expected={str(Path(name).relative_to(P)):value for name,value in expected.items()}
    with tarfile.open(path,'r:gz') as stream:
        members=stream.getmembers()
        need(len(members)==len(expected) and len({m.name for m in members})==len(expected)
            and all(m.isfile() and m.name in expected for m in members),'unique exact source archive coverage')
        for member in members:need(hashlib.sha256(stream.extractfile(member).read()).hexdigest()==expected[member.name],'archive member '+member.name)
    return dict(path=str(path),sha256=sha(path),members=len(expected),all_member_bytes_matched=True)


def module(name):
    if str(P) not in sys.path:sys.path.insert(0,str(P))
    value=importlib.import_module(name);need(Path(value.__file__).resolve()==P/(name+'.py'),'exact readonly module '+name)
    return value


def source(r,tag):
    need(sha(P/'PREPARED_SOURCE_HASHES_V105.json')==FREEZE_SHA and sha(P/'CPU_RESULT_V105.json')==CPU_SHA
        and sha(PRE)==PRE_SHA,'independently reviewed CPU/freeze/prelaunch SHA')
    f=read(P/'PREPARED_SOURCE_HASHES_V105.json');c=read(P/'CPU_RESULT_V105.json');pre=read(PRE)
    need(pre['all_checks_passed'] is True and pre['NPU_started'] is False
        and pre['CPU_sha256']==CPU_SHA and pre['freeze_sha256']==FREEZE_SHA
        and len(f['files'])==pre['static_files']==22 and len(f['external_files'])==pre['external_files']==69,'prelaunch source identities')
    need(c['status']=='passed' and c['tests_run']==f['actual_HOST_tests']==31 and c['failures']==c['errors']==0
        and c['NPU_initialized'] is c['model_initialized'] is False and c['initialization_attempts']==[]
        and c['input_hashes']==c['input_hashes_after']==f['files']
        and c['external_input_hashes']==c['external_input_hashes_after']==f['external_files'],'actual31 HOST source closure')
    expected=f['files']|{str(P/'CPU_RESULT_V105.json'):CPU_SHA,str(P/'PREPARED_SOURCE_HASHES_V105.json'):FREEZE_SHA}
    need(r['source_hashes']==expected and r['external_input_hashes']==f['external_files'],'exact actual launch source maps')
    for name,wanted in (expected|f['external_files']|f['retained_actual_HOST_attempts']).items():need(sha(name)==wanted,'current immutable input '+name)
    need(r['source_snapshot']==str(P/'source-snapshots'/('local-'+tag+'.tar.gz')),'actual per-tag source archive path')
    actual=archive(r['source_snapshot'],expected);need(actual['sha256']==r['source_snapshot_sha256'],'actual archive SHA')
    prepared=archive(pre['archive']['path'],expected|f['retained_actual_HOST_attempts'])
    need(prepared['sha256']==pre['archive']['sha256'] and prepared['members']==27,'independent prepared archive incl retained failures')
    ready=module('check_v105').require_ready();need(ready==r['readiness'] and ready['identity']==pre['readiness_identity'],'original source/HOST readiness')
    shells=r['actual_sourced_runtime_shells_before'];need(shells==r['actual_sourced_runtime_shells_after']
        and {v['path']:v['sha256'] for v in shells}==pre['actual_sourced_runtime_shells_before']
        and all(sha(v['path'])==v['sha256'] for v in shells),'actual retained CANN/ATB shells before/after/current')
    need(r['runtime_aliases_before']==r['runtime_aliases_after'] and all(Path(v['path']).is_symlink()
        and str(Path(v['path']).resolve())==v['resolved'] and Path(v['path']).readlink().as_posix()==v['link']
        and Path(v['path']).exists() for v in r['runtime_aliases_before']),'actual original retained interpreter symlink')
    need(module('source_snapshot_v105').runtime_outputs(P)==r['runtime_outputs'],'actual retained fusion object-pairs diagnostic')
    return dict(static_files=22,external_files=69,actual_HOST_tests=31,CPU_SHA256=CPU_SHA,freeze_SHA256=FREEZE_SHA,
        ROOT_prelaunch_SHA256=PRE_SHA,actual_archive=actual,prepared_archive=prepared,runtime_shells=shells,
        interpreter_aliases=r['runtime_aliases_after'],original_source_contract=ready['source_contract'])


def actual_raw_files(reports,directory):
    files=[]
    for rank,report in enumerate(reports):
        pairs=[(directory/f'rank{rank}-Native-qualification.json.gz',report['qualification'])]
        if rank:pairs+=[(directory/f'rank{rank}-NEW-bank.json.gz',report['bank'])]
        pairs += [(directory/f'rank{rank}-observation{i}.json.gz',row) for i,row in enumerate(report['observations'])]
        for path,expected in pairs:
            raw=path.read_bytes();need(0<len(raw)<16*1024**2 and json.loads(gzip.decompress(raw))==expected,'complete raw gzip/report identity '+path.name)
            files.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
        for record in report['memory']:
            need(record['Native_sticky_nonfinite'] is False and record['reserved']<=54*1024**3
                and record['peak_allocated']<=54*1024**3 and record['free']>=6*1024**3,'actual memory/finite '+str(rank))
        need(report['native_qualification_flags_modified'] is False,'no Native qualification flags assigned')
        if rank:
            builds=report['bank']['prefix_build_calls'];inputs=module('upstream_contract_v105').inputs()
            need(len(builds)==16 and [v['owner'] for v in builds]==list(range(16)),'all16 actual fresh prefix builders')
            need(all(v['input_token_ids']==inputs['prefix'] and v['tokens']==170 and v['positions']==list(range(170))
                and v['sequence_ids']==[v['owner']]*170 and v['use_aclgraph'] is v['use_fused_infer_attention'] is False
                and v['finite'] is v['actual_forward_executed'] is True for v in builds),'real fresh170 eager PA prefix calls')
    return files


def compact(audit):
    comparisons=[]
    for row in audit['comparisons']:
        comparisons.append(dict(round=row['round'],comparison=row['comparison'],same_global_argmax=row['same_global_argmax'],
            hidden=row['target_leader_hidden']['tensors']['row_0'],target_ranks={rank:dict(
                all_saved_tensor_bytes_equal=value['all_tensor_bytes_equal'],
                first_different_root_KV_layer=value['first_different_full_KV_layer'],
                root_local_logits=value['tensors']['root_local_logits'],
                max_root_KV_abs=max(v['max_abs'] for k,v in value['tensors'].items() if k!='root_local_logits'))
                for rank,value in row['target_ranks'].items()}))
    return dict(observations=audit['observations'],comparisons=comparisons,
        scope='15 arm calls=3 sequential A/B/A/C/A rounds on one counterfactual fresh bank, not15 independent requests or4rank trials',
        all64layers_full_root_key_and_value_vectors_checked=True,all16owner_prefixes_hashed_and_query_slots_restored=True,
        query_tensor_diff_scope='owner0 root slot170 only, all64layers and full local3heads×128 dimensions',
        per_arm_MC2_dispatch_counter_snapshots_persisted=False,original_MC2_static_128route_22row_contract_verified=True,
        old_V88_candidate_KV_or_AR_computation_history_reconstructed=False,
        sameM_graph_eager_difference_is_replay_control=True,
        M1_changes_include_M_attention_and_kernel_routes_not_attention_only=True,
        independent_AR_history_qualified=False,performance_gain_proved=False,formal_Goodput=False)


def run(tag):
    need(sys.version_info[:2]==(3,11),'retained Python3.11 required')
    path=P/('receipt-'+tag+'.json');r=read(path);terminal(r,tag);receipt_sha=sha(path);closure=source(r,tag)
    import torch
    torch.set_num_threads(1)
    need(not bool(getattr(getattr(torch,'npu',None),'is_initialized',lambda:False)()),'Torch CPU parser initialized NPU')
    directory=P/'actual'/tag;paths=[directory/f'rank{i}-actual-upstream.json' for i in range(4)]
    need(r['report_paths']==list(map(str,paths)),'exact four actual raw reports')
    reports=[read(p) for p in paths];raw=actual_raw_files(reports,directory)
    audit=module('audit_upstream_v105').audit_worker_reports(reports,private_directory=directory)
    completed=read(directory/'completed.json');replies=[['calibrated',i,dict(rows=0,timings=0)] for i in range(4)]
    need(completed['action']==ACTION and completed['replies']==r['completed_replies']==replies
        and completed['actual_audit']==audit and completed['formal_Goodput'] is False,'independent raw tensor audit matches producer terminal proof')
    need(read(directory/'worker-exit.json')==dict(closed=True,all_exited=True),'actual original workers all exited')
    need(sha(path)==receipt_sha and not bool(getattr(getattr(torch,'npu',None),'is_initialized',lambda:False)()),'stable receipt and CPU-only parser')
    need(not any(name in sys.modules for name in ('vllm_ascend.spec_decode.pearl.native_engine','vllm_ascend.spec_decode.pearl.native_model')),'external audit imported model modules')
    return dict(schema='ROOT.V105.independent-actual-upstream-control-deep-audit.v1',UTC=datetime.now(timezone.utc).isoformat(),
        audit_passed=True,actual_HOST_argv=sys.orig_argv,source_script_SHA256=sha(__file__),
        source_and_HOST_closure=closure,receipt=dict(path=str(path),sha256=receipt_sha,value=r),
        worker_report_files=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in paths],
        actual_raw_gzip_files=raw,independent_actual_tensor_audit=audit,compact_numerical_interpretation=compact(audit),
        CPU_tensor_load='torch.load(map_location=cpu,weights_only=True), SHA and raw BF16 tensor bytes checked before comparison',
        Torch_version=torch.__version__,NPU_initialized=False,model_initialized=False,formal_Goodput=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args();value=run(args.tag)
    path=D/('ROOT_V105_ACTUAL_UPSTREAM_'+args.tag+'_AUDIT.json')
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(path=str(path),sha256=sha(path),audit_passed=True,
        interpretation=value['compact_numerical_interpretation'])))


if __name__=='__main__':main()
