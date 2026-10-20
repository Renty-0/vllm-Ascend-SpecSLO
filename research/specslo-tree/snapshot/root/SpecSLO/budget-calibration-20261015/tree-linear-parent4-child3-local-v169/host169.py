"""Only four new HOST regressions; retained Python3.11, no backend import."""
import ast,importlib.abc,json,sys,time,unittest,subprocess
from base169 import P,sha,require
from check169 import local_files,external_files
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):
   raise ImportError('V169 HOST backend firewall: '+fullname)
def main():
 sys.meta_path.insert(0,Firewall());before=local_files();external=external_files();start=time.perf_counter()
 for path in P.glob('*.py'):ast.parse(path.read_text())
 for path in (P.parent/'budget-diagnostics/v169-root-execution').glob('*.py'):ast.parse(path.read_text())
 for path in (P.parent/'budget-diagnostics/v169-root-execution').glob('*.sh'):subprocess.run(['/bin/bash','-n',str(path)],check=True)
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(P),pattern='test_host169.py'))
 after=local_files();ext_after=external_files();require(before==after and external==ext_after,'source bytes unchanged during HOST')
 require('torch' not in sys.modules and 'torch_npu' not in sys.modules,'no actual backend import')
 value=dict(schema='V169.parent4-local-source-HOST.v1',status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,
  failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.perf_counter()-start,local_before=before,local_after=after,
  external_before=external,external_after=ext_after,actual_interpreter=sys.executable,Python_version=sys.version,
  Torch_imported=False,NPU_used=False,model_constructed=False,freeze_created=False,mathematical_device_qualification=False,formal_Goodput=False)
 path=P/('HOST_ATTEMPT169_'+str(time.time_ns())+'.json')
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
 if result.wasSuccessful():
  cpu=P/'CPU_RESULT169.json';require(not cpu.exists(),'retain previous successful proof')
  with cpu.open('xb') as f:f.write(path.read_bytes())
 print(json.dumps(dict(path=str(path),sha256=sha(path),status=value['status'])))
 return int(not result.wasSuccessful())
if __name__=='__main__':raise SystemExit(main())
