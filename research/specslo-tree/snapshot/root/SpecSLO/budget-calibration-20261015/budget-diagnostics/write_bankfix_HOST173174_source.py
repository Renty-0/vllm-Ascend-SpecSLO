import pathlib
B=pathlib.Path('/root/SpecSLO/budget-calibration-20261015')
test='''"""Actual generated bank lifecycle; explicit HOST model/evidence carriers only."""
import ast,copy,importlib.util,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
from base__E__ import P,BASE,checked,sha,read

class Tests(unittest.TestCase):
 def test_actual_generated_real16_bank_prefix_ready_guard_source_and_original_negative(self):
  import bank__E__ as bank,runtime__E__,trace__E__,base__E__
  self.assertEqual(runtime__E__.P,P);self.assertIs(trace__E__.metrics.__globals__['sha'],base__E__.sha)
  parent=Path(read(P/'INPUTS__E__.json')['bankfix_parent']['path'])
  if str(parent) not in sys.path:sys.path.append(str(parent))
  spec=importlib.util.spec_from_file_location('_frozen_original_bank__O___negative',checked(parent/'bank__O__.py'))
  original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
  cache_path=checked(Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_cache.py'))
  cache_namespace={};exec(compile(cache_path.read_text(),str(cache_path),'exec'),cache_namespace);Cache=cache_namespace['NativePrefixCache']
  class Hidden:
   def clone(self):return Hidden()
  class Engine:
   def __init__(self):
    self.cache=Cache(544,32,128);self.cache_allocation=None;self.is_draft=True
    self.config=types.SimpleNamespace(kvcache_block_size=128);self.calls=[]
   def _allocate_cache(self,prompts,**kwargs):
    self.cache_allocation=self.cache.allocate(prompts,False,sequence_capacity=527);self.cache_block_tables=object()
   def _activate_cache_sequence(self,i,tokens,**kwargs):self.cache.activate_sequence(i,tokens,enable_prefix_caching=False)
   def _ensure_cache_capacity(self,owners,positions):self.cache.ensure_capacity(owners,positions)
   def _run_packed_hidden(self,tokens,owners,positions,**kwargs):
    self.calls.append(dict(tokens=list(tokens),owners=list(owners),positions=list(positions),flags=dict(kwargs)));return Hidden()
  case=bank.cases()[0];e=Engine();expected={i:list(case['owners'][i%16]['token_ids'][:-1]) for i in range(32)};received=[]
  def evidence(engine,histories):
   self.assertIs(engine,e);self.assertEqual(histories,expected);self.assertEqual(list(histories),list(range(32)))
   received.append(copy.deepcopy(histories));return dict(HOST_only=True,exact_history_owners=list(histories))
  torch=types.ModuleType('torch');torch.isfinite=lambda value:types.SimpleNamespace(all=lambda:True)
  self.assertIs(bank.DualBank.prefix_ready.__globals__,vars(bank));self.assertIs(bank.DualBank.guard.__globals__,vars(bank))
  with patch.object(bank,'tensor_hash',lambda value:'explicit-HOST-table-hash'),patch.object(bank,'prefix_evidence',evidence),patch.dict(sys.modules,{'torch':torch}):
   value=bank.build_bank(e,case,stage=lambda name,method:method())
   self.assertEqual(value.page_count,__PAGES__);self.assertEqual(len(value.initial_tables),527)
   self.assertEqual([r['owner'] for r in value.prefill],list(range(32)));self.assertEqual(len(e.calls),32)
   self.assertEqual(len(received),1);self.assertEqual(value.initial_prefix['exact_history_owners'],list(range(32)))
   for i in range(16,32):
    self.assertEqual(e.calls[i]['tokens'],expected[i]);self.assertEqual(e.calls[i]['owners'],[i]*len(expected[i]))
    self.assertEqual(e.calls[i]['positions'],list(range(len(expected[i]))))
   lease=value.guard();self.assertEqual(lease['candidate_owners'],list(range(16)));self.assertEqual(lease['reference_owners'],list(range(16,32)))
   source=value.source();self.assertEqual(source['ids'],list(range(16)))
   for i,row in enumerate(source['contexts']):
    self.assertEqual(row['reference_owner'],i+16);self.assertEqual(row['token_ids'],case['owners'][i]['token_ids'])
    self.assertEqual(row['output_offset'],case['owners'][i].get('output_offset',0))
   # Same actual generated method and real dict; frozen old i%32 must fail at i16.
   with self.assertRaises(IndexError):original.DualBank.prefix_ready(value)
   self.assertEqual(len(received),1)
   saved=e.cache_allocation.block_tables[16][0];e.cache_allocation.block_tables[16][0]=-1
   with self.assertRaises(RuntimeError):value.guard()
   e.cache_allocation.block_tables[16][0]=saved;value.guard()
   value.reference_contexts[0]=value.reference_contexts[0]+[123]
   with self.assertRaises(RuntimeError):value.source()
  self.assertNotIn('torch_npu',sys.modules)
'''
host='''"""One new actual bank HOST regression; previous frozen suites are not rerun."""
import ast,importlib.abc,json,subprocess,sys,time,unittest
from base__E__ import P,sha,require
from check__E__ import local_files,external_files,parent_frozen
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):raise ImportError('V__E__ HOST backend firewall: '+fullname)
def main():
 sys.meta_path.insert(0,Firewall());before=local_files();external=external_files();start=time.perf_counter()
 G=P.parent/'budget-diagnostics/v__E__-root-execution'
 for path in list(P.glob('*.py'))+list(G.glob('*.py')):ast.parse(path.read_text())
 for path in G.glob('*.sh'):subprocess.run(['/bin/bash','-n',str(path)],check=True)
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(P),pattern='test_host__E__.py'))
 after=local_files();ext_after=external_files();require(before==after and external==ext_after,'unchanged exact source across actual bank HOST')
 require('torch' not in sys.modules and 'torch_npu' not in sys.modules,'no actual backend import')
 value=dict(schema='V__E__.real16-bankfix-actual-HOST.v1',status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.perf_counter()-start,local_before=before,local_after=after,external_before=external,external_after=ext_after,frozen_parent=parent_frozen(),actual_interpreter=sys.executable,Python_version=sys.version,Torch_imported=False,NPU_used=False,model_constructed=False,freeze_created=False,model_KV_qualification=False,formal_Goodput=False)
 path=P/('HOST_ATTEMPT__E__PERIOD'+str(time.time_ns())+'.json')
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\\n')
 if result.wasSuccessful():
  cpu=P/'CPU_RESULT__E__.json';require(not cpu.exists(),'retain previous successful HOST proof')
  with cpu.open('xb') as f:f.write(path.read_bytes())
 print(json.dumps(dict(path=str(path),sha256=sha(path),status=value['status'])));return int(not result.wasSuccessful())
if __name__=='__main__':raise SystemExit(main())
'''.replace('PERIOD','_')
for old,new,name,pages in ((167,173,'tree-causal80-real16-bankfix-full-v173',108),(170,174,'tree-normal-cap5-real16-bankfix-full-v174',218)):
 P=B/name
 for filename,source in ((f'test_host{new}.py',test),(f'host{new}.py',host)):
  with (P/filename).open('x') as f:f.write(source.replace('__E__',str(new)).replace('__O__',str(old)).replace('__PAGES__',str(pages)))
