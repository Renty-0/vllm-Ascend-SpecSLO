"""Only the six necessary new HOST checks; no parent suite or model import."""
import ast,importlib.abc,json,sys,time,unittest
from scope152 import P,require,sha
import reuse152
from check152 import local_files,source_contract
class NoBackend(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):raise ImportError('P152 HOST firewall '+name)
def main():
 sys.meta_path.insert(0,NoBackend());before=local_files();external=dict(reuse152.record()['files']);start=time.perf_counter()
 for path in P.glob('*.py'):compile(path.read_text(),str(path),'exec')
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(P),pattern='test_host152.py'))
 contract=source_contract();after=local_files();external_after={p:sha(p) for p in external}
 require(before==after and external==external_after,'actual new HOST own selected source+readonly inputs stable')
 row=dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.perf_counter()-start,
  Torch_imported='torch' in sys.modules,NPU_initialized=False,model_initialized=False,parent_suite_rerun=False,
  input_hashes=before,input_hashes_after=after,external_input_hashes=external,external_input_hashes_after=external_after,source_contract=contract,
  actual_device_qualification=False,formal_Goodput=False,dynamic_prefix_cost_measured=False)
 attempt=P/('HOST_ATTEMPT152_'+str(time.time_ns())+'.json');attempt.write_text(json.dumps(row,indent=2)+'\n')
 if result.wasSuccessful():
  cpu=P/'CPU_RESULT152.json';require(not cpu.exists(),'prior CPU proof retained; use new revision if source changes');cpu.write_bytes(attempt.read_bytes())
 print(json.dumps(dict(path=str(attempt),sha256=sha(attempt),status=row['status'],tests=result.testsRun)))
 return int(not result.wasSuccessful())
if __name__=='__main__':raise SystemExit(main())
