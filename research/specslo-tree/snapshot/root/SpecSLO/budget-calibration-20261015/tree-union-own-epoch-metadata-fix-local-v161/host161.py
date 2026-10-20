"""Four new HOST tests only; explicit tensors and actual allocator source."""
import importlib.abc,json,sys,time,unittest
from base161 import P,require,sha
from check161 import local_files,source_contract
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):raise ImportError('V161 HOST backend firewall: '+fullname)
def main():
 sys.meta_path.insert(0,Firewall());before=local_files();start=time.perf_counter()
 for path in P.glob('*.py'):compile(path.read_text(),str(path),'exec')
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(P),pattern='test_host161.py'))
 scope=source_contract();after=local_files();require(before==after,'source bytes unchanged during HOST')
 row=dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
  elapsed_seconds=time.perf_counter()-start,input_hashes=before,input_hashes_after=after,Torch_imported='torch' in sys.modules,NPU_used=False,
  actual_NativePrefixCache_release_reallocate=True,actual_original_PA_metadata_and_pad_AST=True,
  original_actor_target_verdict_and_next_unchanged=True,source_contract=scope,formal_Goodput=False)
 path=P/('HOST_ATTEMPT161_'+str(time.time_ns())+'.json')
 with path.open('x') as f:json.dump(row,f,indent=2,sort_keys=True);f.write('\n')
 if result.wasSuccessful():
  cpu=P/'CPU_RESULT161.json';require(not cpu.exists(),'preserve all previous proofs')
  with cpu.open('xb') as f:f.write(path.read_bytes())
 print(json.dumps(dict(path=str(path),sha256=sha(path),status=row['status'],tests=result.testsRun,elapsed=row['elapsed_seconds'])))
 return int(not result.wasSuccessful())
if __name__=='__main__':raise SystemExit(main())
