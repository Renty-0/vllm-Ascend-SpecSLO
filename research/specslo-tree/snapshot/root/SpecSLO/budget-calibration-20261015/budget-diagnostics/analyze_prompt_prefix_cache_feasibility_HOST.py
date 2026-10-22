"""Readonly prompt-prefix opportunity and actual pure-cache metadata boundary."""
from pathlib import Path
from collections import Counter,OrderedDict
from datetime import datetime,timezone
import gzip,hashlib,importlib.util,json,sys
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_cache.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p))
record= D/'ROOT_V197_CURRENT_FULL463_RESULT_AND_HANDOFF.json'
r=json.loads(record.read_bytes());raw=Path(r['raw']['path']);assert sha(raw)==r['raw']['sha256']
x=json.loads(gzip.decompress(raw.read_bytes()));prompts=x['tokenized_inputs'];cohorts=r['empty_active_cohort_replay']['cohorts']
assert len(prompts)==463 and sorted(i for c in cohorts for i in c['owners'])==list(range(463))
total=sum(map(len,prompts));counts=Counter(tuple(t)for t in prompts)
ideal={}
for budget in [None,128,256,512,1024]:
 cache=OrderedDict();hits=0
 for cohort in cohorts:
  for i in cohort['owners']:
   ids=prompts[i]
   for n in range(32,len(ids),32):
    key=tuple(ids[:n])
    if key not in cache:break
    cache.move_to_end(key);hits+=32
  # Only publish after all this cohort's prefill. No same-cohort/future hits.
  for i in cohort['owners']:
   ids=prompts[i]
   for n in range(32,len(ids),32):
    key=tuple(ids[:n]);cache[key]=True;cache.move_to_end(key)
    if budget is not None:
     while len(cache)>budget:cache.popitem(last=False)
 ideal[str(budget)]=dict(reused_fullblock_tokens=hits,fraction=hits/total,cached_blocks_at_end=len(cache))
spec=importlib.util.spec_from_file_location('_ROOT_readonly_prefix_cache_metadata',N)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
# A declared tiny HOST allocator history using actual public methods only.
c=module.NativePrefixCache(5,4,2);a=c.allocate([[1,2,3,4,5]],sequence_capacity=3)
old=list(a.block_tables[0]);c.activate_sequence(1,[9,9,0]);c.release_sequence(0)
c.activate_sequence(2,[8,8,0]);c.release_sequence(1)
reported=c.activate_sequence(0,[1,2,3,4,5]);new=list(a.block_tables[0])
assert reported==2 and old[0]!=new[0]and old[1]==new[1]
contiguity=dict(block_size=2,num_blocks=5,blocks_per_sequence=4,sequence_capacity=3,
 actual_api_sequence=['allocate([[1,2,3,4,5]],capacity=3)','activate(1,[9,9,0])','release(0)',
 'activate(2,[8,8,0])','release(1)','activate(0,[1,2,3,4,5])'],
 original_table=old,new_table=new,reported_cached_tokens=reported,
 first_prefix_block_reallocated=True,second_prefix_block_hit=True,
 contiguous_preexisting_cached_prefix_tokens=0,
 metadata_risk='Reported sum of later hits is not necessarily a contiguous already-materialized prefix. Native prefill slices prompt[cached_tokens:], so a future enabled implementation must close this boundary.',
 numerical_KV_or_output_tested=False,Torch_imported=False,NPU_used=False)
assert 'torch'not in sys.modules and 'torch_npu'not in sys.modules
result=dict(schema='ROOT.prompt-prefix-cache-feasibility.HOST.v1',UTC=datetime.now(timezone.utc).isoformat(),
 status='readonly_opportunity_and_metadata_boundary_only',source=pin(N),completed_raw=pin(raw),ROOT_completed_record=pin(record),
 prompts=463,total_prompt_tokens=total,distinct_prompts=len(counts),extra_exact_repeat_requests=sum(n-1 for n in counts.values()),
 original_prefill_cached_tokens_all_zero=all(all(v==0 for v in c['cached_tokens'])for c in x['standalone_prefill_calls']),
 cold_completed_cohort_only_ideal_LRU_model=ideal,actual_pure_cache_contiguity_reproduction=contiguity,
 next_source_requirements=['Do not enable original cache unchanged. Only contiguous published full blocks may count toward skipped prompt prefix; shared blocks must remain immutable and incomplete/tail writes private.',
 'Preserve reference counts, copy-on-write, page-table ownership, actual cached source prefix and all-role completion. New prefix math/root/fullKV/next qualification needed in the actual engine.',
 'Start measurement cache cold; no reuse of warm/qualification requests, future prompts or reference roots. Preserve original arrivals/output/publication and charge all actual production CPU/communication/model costs.',
 'Check actual pool544 occupancy and dummy/reference owners; an independent128-block ideal LRU model is not an exact pooled NativeCache544 execution or a memory-allocation proof.',
 'New cached suffix prefill shapes may require new current MC2/warm/zero qualification. Current full/TP4 baseline and uncached Native assumptions need explicit reporting.'],
 limitations=['No model, tensor/KV comparison, wall timing, cache speedup or GP prediction. Byte-equal past content and formal own-AR are not established.',
 'LRU simulations have no active/decode/page-pressure or reference counting, assume whole blocks and cohort-end publication. Fractions are opportunity bounds for that simplified model, not observed cache hits.',
 'Contiguity example executes only original stdlib allocator. It demonstrates metadata control semantics, not an observed numerical output failure; all existing completed experiments had caching disabled.'],
 private_candidate_created=False,source_files_modified=False,new_cache_enabled=False,models_run=False,NPU_used=False,Torch_imported=False,formal_Goodput=False)
out=D/'ROOT_PROMPT_PREFIX_CACHE_FEASIBILITY_HOST.json'
with out.open('x')as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
print(json.dumps(dict(output=pin(out),prompt_count=463,distinct=len(counts),total_prompt_tokens=total,ideal=ideal,actual_cached_token_metadata=reported)))
