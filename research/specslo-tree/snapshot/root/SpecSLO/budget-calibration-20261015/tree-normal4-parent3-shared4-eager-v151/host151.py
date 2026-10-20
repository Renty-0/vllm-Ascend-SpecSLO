"""Run only the five necessary HOST regressions, preserve every attempt."""
import ast,importlib.abc,json,sys,time,unittest
from scope151 import P,require,sha
from check151 import local_files,source_contract
from originals151 import pins
class NoDevice(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):
   raise ImportError('HOST-only import firewall: '+fullname)
def main():
 sys.meta_path.insert(0,NoDevice());before=local_files();external=dict(pins());start=time.perf_counter()
 for p in P.glob('*.py'):compile(p.read_text(),str(p),'exec')
 suite=unittest.defaultTestLoader.discover(str(P),pattern='test_host151.py');result=unittest.TextTestRunner(verbosity=2).run(suite)
 contract=source_contract();after=local_files();external_after={p:sha(p) for p in external}
 require(before==after and external==external_after,'HOST preserves actual local and parent source bytes')
 row=dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
  elapsed_seconds=time.perf_counter()-start,Torch_imported='torch' in sys.modules,NPU_initialized=False,model_initialized=False,
  input_hashes=before,input_hashes_after=after,external_input_hashes=external,external_input_hashes_after=external_after,
  source_contract=contract,actual_device_qualification=False,formal_Goodput=False)
 path=P/('HOST_ATTEMPT151_'+str(time.time_ns())+'.json');path.write_text(json.dumps(row,indent=2)+'\n')
 print(json.dumps(dict(path=str(path),sha256=sha(path),status=row['status'],tests=result.testsRun)))
 if result.wasSuccessful():
  cpu=P/'CPU_RESULT151.json'
  require(not cpu.exists(),'retain prior CPU receipt; ROOT must choose new revision if sources change')
  cpu.write_bytes(path.read_bytes())
 return int(not result.wasSuccessful())
if __name__=='__main__':raise SystemExit(main())
