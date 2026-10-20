"""Seven new HOST tests, no installed Native method/model/backend."""
import importlib.abc,json,sys,time,unittest,hashlib
from pathlib import Path
from source159 import P,BASE
if str(BASE) not in sys.path:sys.path.insert(1,str(BASE))
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):raise ImportError('V159 HOST backend firewall: '+fullname)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def local():return {str(p):sha(p) for p in sorted(P.iterdir()) if p.is_file() and (p.suffix=='.py' or p.name=='INPUTS159.json')}
def main():
 sys.meta_path.insert(0,Firewall());before=local();start=time.perf_counter()
 for path in P.glob('*.py'):compile(path.read_text(),str(path),'exec')
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(P),pattern='test_host159.py'))
 after=local();assert before==after
 row=dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
  elapsed_seconds=time.perf_counter()-start,input_hashes=before,input_hashes_after=after,
  Torch_imported='torch' in sys.modules,NPU_used=False,installed_on_model_or_device=False,formal_Goodput=False,full_qualification=False)
 path=P/('HOST_ATTEMPT159_'+str(time.time_ns())+'.json')
 with path.open('x') as f:json.dump(row,f,indent=2,sort_keys=True);f.write('\n')
 if result.wasSuccessful():
  cpu=P/'CPU_RESULT159.json';assert not cpu.exists()
  with cpu.open('xb') as f:f.write(path.read_bytes())
 print(json.dumps(dict(path=str(path),sha256=sha(path),status=row['status'],elapsed=row['elapsed_seconds'])))
 return int(not result.wasSuccessful())
if __name__=='__main__':raise SystemExit(main())
