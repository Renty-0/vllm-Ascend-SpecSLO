"""Execute original adapted24/ref24 HOST bank with real Torch CPU allocator."""
import ast,copy,hashlib,importlib.util,json,sys,types,datetime
from pathlib import Path
from typing import Any
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
def need(x,m):
 if not x:raise AssertionError(m)
def main():
 import torch
 need('torch_npu'not in sys.modules,'CPU only')
 sys.path[:0]=[str(D),str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209')];import base209
 from active48_pages_and_bank24_HOST_v2 import bank24_source
 import dual_bank_v146 as old
 from scope_v146 import cases
 case=cases()[0];case=dict(case,owners=[dict(r,reference_owner=i+24)for i,r in enumerate(case['owners'][:24])])
 text,edits=bank24_source(Path(old.__file__).read_text());namespace=dict(vars(old));exec(compile(text,'<uninstalled-HOST24-ref24-bank-v2>','exec'),namespace)
 spec=importlib.util.spec_from_file_location('_HOST24_native_cache_CPU',N/'native_cache.py');cache=importlib.util.module_from_spec(spec);sys.modules[spec.name]=cache;spec.loader.exec_module(cache)
 names={'_allocate_cache','_activate_cache_sequence','_ensure_cache_capacity'};ns=dict(torch=torch,NativeCacheAllocation=cache.NativeCacheAllocation)
 nodes=[copy.deepcopy(n)for n in ast.walk(ast.parse((N/'native_engine.py').read_text()))if isinstance(n,ast.FunctionDef)and n.name in names]
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(N/'native_engine.py')+'::HOST-CPU-original-cache','exec'),ns)
 model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(key_cache=torch.zeros((544,128,1)),value_cache=torch.zeros((544,128,1))))])
 e=types.SimpleNamespace(device=torch.device('cpu'),model=model,prefix_cache=cache.NativePrefixCache(544,32,128),cache_allocation=None,cache_block_tables=None,
  _cache_storage_sequence_capacity=527,config=types.SimpleNamespace(kvcache_block_size=128),is_draft=True)
 for name in names:setattr(e,name,types.MethodType(ns[name],e))
 # Fake hidden production is explicit; only allocator/ownership/cache indexing is tested.
 e._run_packed_hidden=lambda tokens,*a,**kw:torch.zeros((len(tokens),1))
 bank=namespace['build_bank'](e,case,stage=lambda label,fn:fn())
 need(bank.page_count==160 and bank.guard()['candidate_owners']==list(range(24))and bank.guard()['reference_owners']==list(range(24,48)),'24/ref24 real ownership and160 pages')
 need(len(bank.prefill)==48 and len(bank.initial_prefix['KV'])==2,'all48 prefix builds reach actual prefix_evidence indexing')
 need(len(bank.source()['ids'])==24 and set(bank.reference_ledger)==set(range(24,48)),'all24 distinct reference states')
 need('torch_npu'not in sys.modules,'no NPU imported')
 result=dict(status='passed_HOST_CPU_only',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_bank_class_and_builder_executed=True,
  real_original_allocator_methods=True,independent_prefix_builds=48,candidate_owners=24,reference_owners=24,pages=160,scope='HOST owner/page/modulo geometry; one fake CPU layer and fake hidden; no neural/latency qualification',
  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in[Path(__file__),D/'active48_pages_and_bank24_HOST_v2.py',Path(old.__file__),N/'native_cache.py',N/'native_engine.py']},Torch_CPU=True,NPU_used=False,installed=False,adopted=False)
 p=D/'ROOT_ACTIVE24_REF24_BANK_V2_REAL_TORCH_CPU.json';need(not p.exists(),'fresh HOST evidence');p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='source_sha256'}))
if __name__=='__main__':main()
