"""Compare cached and original transforms with changed live inputs, CPU only."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import dataclasses,hashlib,importlib.util,json,sys,time
from pathlib import Path
import torch
torch.set_num_threads(1)
from projection247 import transform,CACHE,ORIGINAL,IndexCache

spec=importlib.util.spec_from_file_location('_original243_for_cache_HOST',ORIGINAL)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)

@dataclasses.dataclass
class Meta:
    slot_mapping:object;context_lens:object;block_tables:object;request_block_tables:object
    tree_attention_mask:object;actual_seq_lengths_q:object;sequence_lens:object
    tree_attention:bool=True;use_fused_infer_attention:bool=True


def compare(a,b):
    for left,right in zip(a[:2]+a[3:4],b[:2]+b[3:4],strict=True):assert torch.equal(left,right)
    for f in dataclasses.fields(Meta):
        left,right=getattr(a[2],f.name),getattr(b[2],f.name)
        assert torch.equal(left,right)if isinstance(left,torch.Tensor)else left==right
    assert a[4]==b[4]


def frame(case,variant,mask_capacity):
    sizes=[]
    for b in case['budgets']:sizes.extend([5,1]if b==5 else[b+1])
    sizes.extend(case['padding']);ends=[];cursor=0
    for size in sizes:cursor+=size;ends.append(cursor)
    assert cursor==80
    groups=len(sizes);mask=torch.ones((groups,1,9,4096),dtype=torch.bool)
    for group,size in enumerate(sizes):
        for q in range(size):mask[group,0,q,:1+(group*13+q*7+variant*71)%2550]=False
    tables=(torch.arange(groups*32,dtype=torch.int32).reshape(groups,32)+variant*5)%543
    tables[:,20:]=-1
    meta=Meta(torch.arange(80,dtype=torch.long)+variant*1000,
        torch.arange(80,dtype=torch.int32)+variant,tables,tables,mask,tuple(ends),(2550,)*groups)
    return(torch.arange(80,dtype=torch.long)+variant*17,torch.arange(80,dtype=torch.long)+variant*31,
           meta,case['query_count'],526),dict(mask_capacity=mask_capacity)


def main():
    proof=Path('/root/SpecSLO/budget-calibration-20261015/tree-causal80-small-cohort-role-aware-audit-full-v209/actual/static-q5-root-v245-full463-ownAR-v1/canonical-independent-node-math/rank1-proof.json')
    cases=[row['case']for row in json.loads(proof.read_text())['math']['cases']]
    assert len(cases)==111
    checked=0;fresh_dynamic_tensors=0
    for case in cases:
        capacity=11 if 5 in case['budgets']else 10
        for variant in (0,1,0):
            args,kwargs=frame(case,variant,capacity)
            actual=transform(*args,**kwargs);expected=original.transform(*args,**kwargs)
            compare(actual,expected);checked+=1
            # Changing live values never changes only a cache key: recomputation
            # must produce fresh tables/masks and actual changed inputs/slots.
            again=transform(*args,**kwargs)
            compare(again,expected)
            assert actual[2].tree_attention_mask.data_ptr()!=again[2].tree_attention_mask.data_ptr()
            assert actual[2].request_block_tables.data_ptr()!=again[2].request_block_tables.data_ptr()
            fresh_dynamic_tensors+=2
    # A different original mask row stride must not reuse incorrect source rows.
    args,kwargs=frame(cases[0],2,10)
    args=list(args);meta=args[2]
    meta.tree_attention_mask=torch.cat([meta.tree_attention_mask,torch.ones_like(meta.tree_attention_mask[:,:,:1])],dim=2)
    compare(transform(*args,**kwargs),original.transform(*args,**kwargs));checked+=1
    # Explicit bounded eviction and immutable entries remain value-correct.
    local=IndexCache(maximum=1)
    p=original.plan((5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80),5)
    first=local.get(p,torch.device('cpu'),9,10)
    other=local.get(p,torch.device('cpu'),10,11)
    restored=local.get(p,torch.device('cpu'),9,10)
    assert local.evictions==2 and local.misses==3 and len(local.entries)==1
    assert all(torch.equal(first[k],restored[k])for k in first)
    assert 'torch_npu'not in sys.modules
    result=dict(status='HOST_cached_projection_equals_original_with_live_changes',
        supported_cases=111,changed_restored_and_stride_comparisons=checked,
        verified_fresh_dynamic_mask_and_table_tensors=fresh_dynamic_tensors,
        cache_hits=CACHE.hits,cache_misses=CACHE.misses,bounded_entries=len(CACHE.entries),
        eviction_negative_control=True,original_source_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
        candidate_source_sha256=hashlib.sha256(Path(__file__).with_name('projection247.py').read_bytes()).hexdigest(),
        fixture_proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),
        NPU_used=False,neural_qualification=False,performance_qualification=False,production_installed=False)
    output=Path(__file__).with_name('HOST_CACHED_PROJECTION247_RESULT.json')
    with output.open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
