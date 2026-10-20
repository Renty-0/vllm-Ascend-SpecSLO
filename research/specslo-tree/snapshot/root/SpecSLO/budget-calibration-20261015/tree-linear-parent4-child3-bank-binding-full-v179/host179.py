"""Three necessary lifecycle/binding/guard checks; no old suite rerun."""
import ast,importlib.abc,json,subprocess,sys,time,unittest
from base179 import P,sha,require
from check179 import local_files,external_files
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):raise ImportError('V179 HOST backend firewall '+name)
def main():
 sys.meta_path.insert(0,Firewall());before=local_files();external=external_files();start=time.perf_counter()
 for p in P.glob('*.py'):ast.parse(p.read_text())
 for p in (P.parent/'budget-diagnostics/v179-root-execution').glob('*.py'):ast.parse(p.read_text())
 for p in (P.parent/'budget-diagnostics/v179-root-execution').glob('*.sh'):subprocess.run(['/bin/bash','-n',str(p)],check=True)
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(P),pattern='test_host179.py'))
 after=local_files();extafter=external_files();require(before==after and external==extafter,'actual source pre/post')
 require(not any(n.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for n in sys.modules),'HOST carriers removed; no actual backend import')
 row=dict(schema='V179.actual-bank-lifecycle-source-HOST.v1',status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,
  failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.perf_counter()-start,local_before=before,local_after=after,
  external_before=external,external_after=extafter,actual_interpreter=sys.executable,Python_version=sys.version,
  Torch_imported=False,NPU_used=False,HOST_tensor_carriers_used=True,actual_NativePrefixCache_used=True,actual_build_bank_used=True,
  real_model_or_device_math_tested=False,old4HOST_suite_rerun=False,freeze_created=False,parent_actual_qualification_inherited=False,formal_Goodput=False)
 p=P/('HOST_ATTEMPT179_'+str(time.time_ns())+'.json')
 with p.open('x') as f:json.dump(row,f,indent=2,sort_keys=True);f.write('\n')
 if result.wasSuccessful():
  cpu=P/'CPU_RESULT179.json';require(not cpu.exists(),'preserve all earlier proof bytes')
  with cpu.open('xb') as f:f.write(p.read_bytes())
 print(json.dumps(dict(path=str(p),sha256=sha(p),status=row['status'])))
 return int(not result.wasSuccessful())
if __name__=='__main__':raise SystemExit(main())
