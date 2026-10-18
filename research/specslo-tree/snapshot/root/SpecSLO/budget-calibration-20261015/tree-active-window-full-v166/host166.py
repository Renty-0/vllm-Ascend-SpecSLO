"""Only five new HOST checks, with backend import firewall and byte retention."""
import ast,importlib.abc,json,subprocess,sys,time,unittest
from base166 import P,B,require,sha
from check166 import local_files,external_files
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):
   raise ImportError('V166 HOST backend firewall: '+fullname)
def main():
 sys.meta_path.insert(0,Firewall());before=local_files();external=external_files();start=time.perf_counter()
 for path in P.glob('*.py'):ast.parse(path.read_text())
 G=B/'budget-diagnostics/v166-root-execution'
 for path in G.glob('*.sh'):subprocess.run(['/bin/bash','-n',str(path)],check=True)
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(P),pattern='test_host166.py'))
 after=local_files();ext_after=external_files();require(before==after and external==ext_after,'source bytes unchanged during HOST')
 require('torch' not in sys.modules and 'torch_npu' not in sys.modules,'no actual backend import')
 value=dict(schema='V166.active-window-full-source-HOST.v1',status='passed' if result.wasSuccessful() else 'failed',
  tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.perf_counter()-start,
  local_before=before,local_after=after,external_before=external,external_after=ext_after,
  Torch_imported=False,NPU_used=False,model_constructed=False,freeze_created=False,
  private_IO_AST_executed=True,actual_new_shell_and_installer_with_HOST_stubs=True,
  actual_fixedpoint_loop_with_stubs=True,actual_guard_constructor_AST_and_duplicate_negative=True,
  original_parent_suite_rerun=False,full_measured=False,formal_Goodput=False)
 path=P/('HOST_ATTEMPT166_'+str(time.time_ns())+'.json')
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
 if result.wasSuccessful():
  cpu=P/'CPU_RESULT166.json';require(not cpu.exists(),'preserve any prior success proof')
  with cpu.open('xb') as f:f.write(path.read_bytes())
 print(json.dumps(dict(path=str(path),sha256=sha(path),status=value['status'])))
 return int(not result.wasSuccessful())
if __name__=='__main__':raise SystemExit(main())
