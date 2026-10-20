"""Six necessary HOST checks with backend import firewall and retained attempts."""
import ast,importlib.abc,json,sys,time,unittest
from scope154 import P,require,sha
from check154 import local_files,source_contract
from reuse154 import inputs
class NoDevice(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):
   raise ImportError('V154 HOST-only import firewall: '+fullname)
def main():
 sys.meta_path.insert(0,NoDevice());before=local_files();external=dict(inputs()['files']);start=time.perf_counter()
 for p in P.glob('*.py'):compile(p.read_text(),str(p),'exec')
 suite=unittest.defaultTestLoader.discover(str(P),pattern='test_host154.py')
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 contract=source_contract();after=local_files();external_after={p:sha(p) for p in external}
 require(before==after and external==external_after,'actual HOST preserved all local and read-only source bytes')
 row=dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,
  failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.perf_counter()-start,
  Torch_imported='torch' in sys.modules,NPU_initialized=False,model_initialized=False,
  input_hashes=before,input_hashes_after=after,external_input_hashes=external,external_input_hashes_after=external_after,
  source_contract=contract,actual_device_qualification=False,formal_Goodput=False)
 path=P/('HOST_ATTEMPT154_'+str(time.time_ns())+'.json')
 with path.open('x') as f:json.dump(row,f,indent=2);f.write('\n')
 print(json.dumps(dict(path=str(path),sha256=sha(path),status=row['status'],tests=result.testsRun)))
 if result.wasSuccessful():
  cpu=P/'CPU_RESULT154.json';require(not cpu.exists(),'retain earlier CPU proof; new sources require explicit revision')
  with cpu.open('xb') as f:f.write(path.read_bytes())
 return int(not result.wasSuccessful())
if __name__=='__main__':raise SystemExit(main())
