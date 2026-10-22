"""Read-only V129 final entry/archive audit. Never executes a valid supervisor."""
from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, os, re, subprocess, sys, tarfile, traceback
from datetime import datetime,timezone
from pathlib import Path

B=Path('/root/SpecSLO/budget-calibration-20261015');P=B/'tree-s35-serialized-evidence-v129';D=B/'budget-diagnostics'
CPU_SHA='48f220196ec5922d528c24eadfc4482db287234a2009f0a191555f7f2ac75478'
FREEZE_SHA='c73e1d22d6cb5e364b415bc29e820ba6f520b8b49b41ec47dc7bf3ef629f92ca'
TAR_SHA='4e1237f5f0300f1ec82793946a9d6f222998c143f24ef76ffff470b3b0a97b15'
PROOF_SHA='53ab60c4412c8d3493bd7cbffc5285e5dd88deb1e6edacf6b01cf0d09e4a6bea'
TAG='s35-serialized-evidence-paired-full463-v129-1'
PY='/root/miniconda3/envs/vllm-hust-dev/bin/python'

def need(condition,message):
    if not condition:raise ValueError(message)

def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):result.update(block)
    return result.hexdigest()

def desc(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)

def read(path):return json.loads(Path(path).read_bytes())

def run():
    before={str(q):sha(q) for q in P.iterdir() if q.is_file()}
    cpu_path=P/'CPU_RESULT_V129.json';frozen_path=P/'PREPARED_SOURCE_HASHES_V129.json'
    need(sha(cpu_path)==CPU_SHA and sha(frozen_path)==FREEZE_SHA,'Pinned V129 CPU/freeze')
    cpu=read(cpu_path);frozen=read(frozen_path)
    need(frozen['CPU_sha256']==CPU_SHA and frozen['CPU_tests_run']==16
        and frozen['ancestor_actual_HOST_tests']==187 and frozen['new_actual_HOST_tests']==16,'New16/ancestor187 identity')
    need(cpu['status']=='passed' and cpu['tests_run']==16 and cpu['failures']==cpu['errors']==0
        and cpu['initialization_attempts']==[] and cpu['NPU_initialized'] is cpu['model_initialized'] is False
        and cpu['torch_imported'] is cpu['torch_npu_imported'] is False and cpu['runtime_wrapper_invoked'] is False
        and cpu['input_hashes']==cpu['input_hashes_after']==frozen['files']
        and cpu['external_input_hashes']==cpu['external_input_hashes_after']==frozen['external_files']
        and cpu['parent_187_HOST_binding']==cpu['parent_187_HOST_binding_after'],'Actual final pure HOST result')
    transcript=cpu['transcript'];need(len(re.findall(r'^test_.* \.\.\. ok$',transcript,re.M))==16
        and 'Ran 16 tests' in transcript and transcript.rstrip().endswith('OK'),'Actual16 transcript')
    for path in set(frozen['files'])&set(frozen['external_files']):
        need(frozen['files'][path]==frozen['external_files'][path],'Conflicting local/external identity')
    expected={**frozen['files'],**frozen['external_files'],str(cpu_path):CPU_SHA,str(frozen_path):FREEZE_SHA}
    for path,wanted in expected.items():need(sha(path)==wanted,'Frozen source byte mismatch: '+path)
    parent=read(P/'V129_PARENT_SOURCE.json');ancestor=read(parent['CPU']);oldfreeze=read(parent['freeze'])
    binding=cpu['parent_187_HOST_binding']
    need(binding['tests_run']==187 and binding['all_original_inputs_rechecked'] is True
        and binding['ancestor_tests_rerun_in_new_epoch'] is False
        and binding['CPU']==parent['CPU'] and binding['CPU_sha256']==parent['CPU_sha256']
        and sha(parent['CPU'])==parent['CPU_sha256'] and sha(parent['freeze'])==parent['freeze_sha256']
        and ancestor['status']=='passed' and ancestor['tests_run']==187
        and ancestor['failures']==ancestor['errors']==0
        and ancestor['input_hashes']==ancestor['input_hashes_after']==oldfreeze['files']==parent['files']
        and ancestor['external_input_hashes']==ancestor['external_input_hashes_after']==oldfreeze['external_files'],
        'Ancestor187 was byte-bound, not rerun')
    for path,wanted in {**oldfreeze['files'],**oldfreeze['external_files']}.items():need(sha(path)==wanted,'Ancestor187 source byte mismatch')
    delta=read(P/'V129_SOURCE_DELTA.json')
    changed={Path(path).name:sha(P/Path(path).name) for path,wanted in parent['files'].items() if sha(P/Path(path).name)!=wanted}
    allowed={'shape_scope_v101.py','combined_runtime_v113.py','audit_combined_v113.py','launch_combined_v113.sh',
        'combined-startup-body.sh','check_v113.py','source_snapshot_v113.py','STATIC_INPUTS_V113.json','COMBINATION_V113.json'}
    need(changed==delta['changed_copied_files'] and set(changed)==allowed,'Exact copied-source delta')
    for name in ('bounded_report_audit_v78.py','boundary_schema_v101.py','native_boundaries.py','original_hook_adapter.py',
        'production_probe_v101.py','local_qualification_scope.py','local_mixed_qualification.py','independent_history_v75.py',
        'normal_draft_actor_v75.py','combined_controls_v113.py','budget_policy.py','startup_config_v75.py'):
        need((P/name).read_bytes()==(Path(parent['directory'])/name).read_bytes(),'Original Native/validator/algorithm changed')
    archive=P/'host-archive/V129_FINAL_CPU_SOURCE.tar.gz';need(sha(archive)==TAR_SHA,'Pinned full archive SHA')
    names={path.lstrip('/'):wanted for path,wanted in expected.items()};need(len(names)==len(expected),'Archive path collision')
    with tarfile.open(archive,'r:gz') as source:
        members=source.getmembers()
        need(len(members)==len(names)==3932 and {m.name for m in members}==set(names)
            and all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members),
            'Exact3932 regular archive members')
        for member in members:need(hashlib.sha256(source.extractfile(member).read()).hexdigest()==names[member.name],'Tar member bytes: '+member.name)
    manifest=read(P/'host-archive/V129_FINAL_CPU_SOURCE.json')
    need(manifest['members']==names and manifest['sha256']==TAR_SHA and manifest['bytes']==archive.stat().st_size,'Archive manifest identity')
    config=read(P/'COMBINATION_V113.json');proof=config['full_pipeline_factory_source_proof']
    need(proof['sha256']==PROOF_SHA and sha(proof['path'])==PROOF_SHA,'V129 full-target factory caller proof')
    factory=read(proof['path']);need(factory['source_dataflow_closed_for_V129_actual_target_pipeline'] is True
        and factory['global_factory_equivalence_proved'] is False,'Factory proof scope')
    for path,wanted in factory['source_sha256'].items():need(expected.get(path)==wanted and sha(path)==wanted,'Factory input absent from source archive')
    # Actual missing-action argv is safe: argparse exits before imports/device queries.
    supervisor=P/'supervise_combined_v113.py'
    bad_argv=[PY,'-B',str(supervisor),'--tag',TAG]
    bad=subprocess.run(bad_argv,cwd=P,capture_output=True,text=True,timeout=15)
    need(bad.returncode==2 and 'required: --action' in bad.stderr,'Actual omitted action must be rejected by argparse')
    need(not (P/f'local-receipt-{TAG}.json').exists() and not (P/'local-output'/TAG).exists(),'Audit must not create an actual run')
    # Replay only the actual parser statements. Never run the valid full main.
    tree=ast.parse(supervisor.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    parser_statements=main.body[:4]
    need(isinstance(parser_statements[-1],ast.Assign) and isinstance(parser_statements[-1].value,ast.Call)
        and isinstance(parser_statements[-1].value.func,ast.Attribute) and parser_statements[-1].value.func.attr=='parse_args','Exact actual parser boundary')
    namespace={'argparse':argparse};previous=sys.argv
    correct=[PY,'-B',str(supervisor),'--action','combined-full463','--tag',TAG]
    try:
        sys.argv=[str(supervisor),*correct[3:]]
        exec(compile(ast.Module(body=parser_statements,type_ignores=[]),str(supervisor),'exec'),namespace)
    finally:sys.argv=previous
    need(vars(namespace['args'])==dict(tag=TAG,action='combined-full463'),'Corrected argv actual parser AST')
    # Run original outer->inner shell guards/argv, stubbing only env/runtime and Python binary boundaries.
    helper=P/'test_launcher_v119_host.py';spec=importlib.util.spec_from_file_location('_v129_frozen_shell_stub',helper)
    stubs=importlib.util.module_from_spec(spec);spec.loader.exec_module(stubs)
    positive=stubs.exercise('combined-full463','HOST-independent-final-v129')
    need(positive['exit_code']==0 and len(positive['tool_argv'])==3,'Actual two-shell stub entry')
    argv=[r['argv'] for r in positive['tool_argv']]
    need(argv[0]==['-B',str(P/'check_v113.py'),'--require-local','combined-full463'],'Final actual readiness argv path')
    need(argv[1]==['-B',str(B/'serial-v1/map-current.py'),'2,5,6,7'],'Actual fresh-mapping command path; not queried here')
    repo=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust')
    extension=repo/'vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so'
    final=['-B','-u',str(repo/'examples/run_with_vllm_ascend_extension.py'),'--extension',str(extension),
        '--required-op','allreduce_add_rmsnorm_chained',str(P/'combined_runtime_v113.py'),
        '--action','combined-full463','--tag','HOST-independent-final-v129']
    need(argv[2]==final,'Final runtime extension argv')
    paths=[supervisor,P/'launch_combined_v113.sh',P/'combined-startup-body.sh',P/'check_v113.py',P/'combined_runtime_v113.py',
        B/'serial-v1/runtime.sh',B/'serial-v1/map-current.py',repo/'examples/run_with_vllm_ascend_extension.py',extension,Path(PY)]
    for path in paths:need(path.is_file(),'Final argv path missing: '+str(path))
    for name in ('launch_combined_v113.sh','combined-startup-body.sh'):
        roots=re.findall(r'^task_root=(.+)$',(P/name).read_text(),re.M)
        need(roots==[str(P)],'Actual shell task_root must be exact V129')
        grammar=subprocess.run(['bash','-n',str(P/name)],capture_output=True,text=True,timeout=15)
        need(grammar.returncode==0,'Actual shell grammar')
    negative=[]
    for inner in (False,True):
        for action,tag in (('s35-qualify65','HOST-old'),('s35-full463','HOST-old'),('combined-full463','bad/tag')):
            result=stubs.exercise(action,tag,inner=inner)
            need(result['exit_code']==2 and result['tool_argv']==[],'Original guard must reject before any tool boundary')
            negative.append(result)
    after={str(q):sha(q) for q in P.iterdir() if q.is_file()}
    need(before==after,'Frozen V129 root file bytes changed during HOST audit')
    need('torch' not in sys.modules and 'torch_npu' not in sys.modules,'HOST audit imported Torch')
    return dict(status='passed',corrected_ROOT_launch_argv=correct,previous_advertised_command_omitted_required_action=True,
        actual_omitted_action_rejection=dict(argv=bad_argv,returncode=bad.returncode,stdout=bad.stdout,stderr=bad.stderr),
        corrected_valid_supervisor_executed=False,actual_argparse_AST_only_replayed=True,
        original_two_shell_stub_execution=positive,original_guard_negative_executions=negative,
        final_extension_python_binary=str(PY),actual_final_extension_argv=argv[2],
        final_entry_paths=[desc(p) for p in paths],real_runtime_wrapper_or_extension_executed=False,
        actual_mapping_query_executed=False,NPU_query_executed=False,NPU_initialized=False,model_initialized=False,
        CPU=desc(cpu_path),freeze=desc(frozen_path),new_actual_HOST_tests=16,new_CPU_elapsed_seconds=cpu['elapsed_seconds'],
        ancestor_actual_HOST_tests=187,ancestor187_rerun_in_V129=False,ancestor_all273_3508_original_inputs_rechecked=True,
        current_local_source_count=len(frozen['files']),current_external_source_count=len(frozen['external_files']),
        full_archive={**desc(archive), 'members':len(names),'all_member_bytes_independently_recomputed':True},
        full_factory_proof=desc(proof['path']),frozen_source_before_after_equal=True,
        actual_device_structural_or_performance_qualification_granted=False,formal_Goodput=False,scope='HOST entry/source/archive readiness only')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    start=datetime.now(timezone.utc).isoformat()
    try:result=run();result['errors']=[]
    except BaseException:result=dict(status='failed',errors=[traceback.format_exc()],NPU_initialized=False,model_initialized=False)
    result.update(schema='V129.independent-final-entry-source-archive-HOST-audit.v1',started_UTC=start,
        finished_UTC=datetime.now(timezone.utc).isoformat(),actual_audit_argv=sys.argv,actual_python=sys.executable,
        runtime_wrapper_used=False,auditor_source=desc(__file__))
    with Path(args.output).open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','errors')}|dict(output=args.output)))
    return int(result['status']!='passed')

if __name__=='__main__':raise SystemExit(main())
