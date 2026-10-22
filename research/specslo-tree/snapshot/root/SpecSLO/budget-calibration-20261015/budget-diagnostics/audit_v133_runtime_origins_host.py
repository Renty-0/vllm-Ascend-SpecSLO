"""Actual runpy entry with the real examples-first import order; stop before model."""
import hashlib,importlib,importlib.abc,json,runpy,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
P=Path('/root/SpecSLO/budget-calibration-20261015/tree-kernel-verdict-repeatfix-v133');D=P.parent/'budget-diagnostics'
REPO=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust');CORE=REPO.parent/'vllm-hust'
NAMES=('check_v125','kernel_verdict_contract_v125','linear_mixed_contract_v117','launch_epoch_v133','kernel_verdict_actor_v125','kernel_verdict_orchestration_v125','linear_mixed_actor_v117','linear_mixed_metadata_v117','tree_target_replay','target_capture','audit_kernel_verdict_v125','audit_linear_mixed_v117','fixed_mask_extent','mc2_inventory_contract')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
class NoModels(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST origin audit forbids '+fullname)
class StopBeforeDirectoryOrModel(BaseException):pass

def main():
 start=datetime.now(timezone.utc).isoformat();result={};guard=NoModels();sys.meta_path.insert(0,guard)
 try:
  if sys.version_info[:2]!=(3,11):raise RuntimeError('retained3.11 required')
  if any(name in sys.modules for name in NAMES):raise RuntimeError('fresh interpreter helper namespace required')
  frozen=json.loads((P/'PREPARED_SOURCE_HASHES_V125.json').read_text());before={n:sha(n) for n in frozen['files']}
  original=list(sys.path)
  # Actual extension script is argv[0], local root follows via exact shell PYTHONPATH.
  sys.path[:]=[str(REPO/'examples'),str(P),str(REPO),str(CORE),*original[1:]]
  module=importlib.import_module('check_v125');actual=module.require_ready;calls=[]
  def stop(*args,**kwargs):
   calls.append(actual(*args,**kwargs));raise StopBeforeDirectoryOrModel()
  module.require_ready=stop
  runtime=P/'kernel_verdict_runtime_v125.py';args=[str(runtime),'--action','kernel-verdict-teacher-PA-control','--tag','HOST-runpy-origin-v133']
  sys.argv=args
  try:runpy.run_path(str(runtime),run_name='__main__')
  except StopBeforeDirectoryOrModel:pass
  else:raise RuntimeError('actual runpy did not stop at post-readiness boundary')
  if len(calls)!=1 or (P/'actual/HOST-runpy-origin-v133').exists():raise RuntimeError('HOST stop boundary changed')
  module.require_ready=actual
  origins=[]
  for name in NAMES:
   item=importlib.import_module(name);path=Path(item.__file__).resolve();expected=P/(name+'.py')
   if path!=expected or frozen['files'].get(str(path))!=sha(path):raise RuntimeError('actual imported origin differs: '+name+' '+str(path))
   functions={k:v.__code__.co_filename for k,v in vars(item).items() if callable(v) and getattr(v,'__module__',None)==name and hasattr(v,'__code__')}
   if any(Path(value).resolve()!=expected for value in functions.values()):raise RuntimeError('actual function origin differs: '+name)
   origins.append(dict(module=name,actual_file=str(path),sha256=sha(path),actual_local_function_code_files=sorted(set(functions.values()))))
  if before!={n:sha(n) for n in frozen['files']} or any(n in sys.modules for n in ('torch','torch_npu','vllm','vllm_ascend')):raise RuntimeError('source changed or forbidden import')
  result=dict(status='passed',actual_runpy_runtime=str(runtime),actual_script_argv=args,sys_path_prefix=sys.path[:4],actual_imported_origins=origins,actual_original_require_ready_completed=True,
   actual_runtime_main_executed_until_post_readiness=True,stopped_before_output_directory_and_model=True,cwd_was_local_candidate=False,actual_NPU_worker_import_origins_observed=False,source_before_after_equal=True,
   exact_source_epoch_required='V133',frozen_source_count=len(frozen['files']),NPU_initialized=False,NPU_query_executed=False,model_initialized=False,errors=[])
 except BaseException:result=dict(status='failed',errors=[traceback.format_exc()],NPU_initialized=False)
 finally:sys.meta_path.remove(guard)
 result.update(schema='V133.actual-examples-first-runpy-helper-origin-HOST-audit.v1',started_UTC=start,finished_UTC=datetime.now(timezone.utc).isoformat(),actual_python=sys.executable,runtime_wrapper_used=False,auditor=dict(path=__file__,sha256=sha(__file__)))
 output=D/'V133_ACTUAL_RUNPY_HELPER_ORIGINS_HOST_AUDIT.json'
 with output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
 print(json.dumps(dict(status=result['status'],output=str(output),sha256=sha(output),errors=result['errors'])));return int(result['status']!='passed')
if __name__=='__main__':raise SystemExit(main())
