"""New seven HOST checks only; retain every attempt and selected source snapshot."""
import ast,importlib.abc,io,json,subprocess,sys,time,unittest
from pathlib import Path
from base205 import P,B,read,sha,require
from check205 import local_files,external_files
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):raise ImportError('V205 HOST forbids backend '+name)
def main():
 sys.meta_path.insert(0,Firewall());start=time.perf_counter();local=local_files();external=external_files()
 attempt=P/'full-host-attempts'/str(time.time_ns());attempt.mkdir(parents=True,exist_ok=False);sources=attempt/'sources';sources.mkdir()
 G=B/'budget-diagnostics/v205-root-execution'
 for path in [*[Path(p) for p in local],*sorted(G.iterdir())]:
  if path.is_file(): (sources/(('guard-' if path.parent==G else '')+path.name)).write_bytes(path.read_bytes())
 for p in P.glob('*.py'):compile(ast.parse(p.read_text()),str(p),'exec')
 for p in G.glob('*.sh'):subprocess.run(['/bin/bash','-n',str(p)],check=True)
 from test_full205 import Tests,DETAILS
 log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 after=local_files();extafter=external_files();require(local==after and external==extafter,'selected source before/after exact')
 require(not any(n.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for n in sys.modules),'outer HOST has no backend')
 (attempt/'unittest.log').write_text(log.getvalue())
 row=dict(schema='V205.full-source-HOST.v1',status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.perf_counter()-start,
  local_before=local,local_after=after,external_before=external,external_after=extafter,details=DETAILS,Torch_imported=False,child_Torch_CPU_control_only=False,NPU_used=False,model_constructed=False,old_suites_rerun=False,freeze_created=False,
  model_math_qualified=False,full463_executed=False,attempt=str(attempt),log_sha256=sha(attempt/'unittest.log'))
 path=P/('HOST_ATTEMPT205_'+str(time.time_ns())+'.json');path.write_text(json.dumps(row,indent=2,sort_keys=True)+'\n');(attempt/'RESULT.json').write_bytes(path.read_bytes())
 if result.wasSuccessful():
  cpu=P/'CPU_RESULT205.json';require(not cpu.exists(),'preserve prior success');cpu.write_bytes(path.read_bytes())
 print(log.getvalue());print(json.dumps(dict(path=str(path),sha256=sha(path),status=row['status'])));return int(not result.wasSuccessful())
if __name__=='__main__':raise SystemExit(main())
