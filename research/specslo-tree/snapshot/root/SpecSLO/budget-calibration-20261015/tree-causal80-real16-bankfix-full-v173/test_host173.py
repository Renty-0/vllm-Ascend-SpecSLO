"""Actual generated bank lifecycle; explicit HOST model/evidence carriers only."""
import ast,copy,importlib.util,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
from base173 import P,BASE,checked,sha,read

class Tests(unittest.TestCase):
 def test_actual_generated_real16_bank_prefix_ready_guard_source_and_original_negative(self):
  import bank173 as bank,runtime173,trace173,base173
  self.assertEqual(runtime173.P,P);self.assertIs(trace173.metrics.__globals__['sha'],base173.sha)
  parent=Path(read(P/'INPUTS173.json')['bankfix_parent']['path'])
  if str(parent) not in sys.path:sys.path.append(str(parent))
  spec=importlib.util.spec_from_file_location('_frozen_original_bank167_negative',checked(parent/'bank167.py'))
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
   self.assertEqual(value.page_count,108);self.assertEqual(len(value.initial_tables),527)
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
