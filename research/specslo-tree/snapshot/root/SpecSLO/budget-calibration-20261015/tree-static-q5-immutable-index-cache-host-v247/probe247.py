"""NPU metadata-only paired timing. No model, attention, KV cache or decoder."""
import dataclasses,hashlib,importlib.util,json,statistics,sys,time
from pathlib import Path
import torch
import torch_npu
from projection247 import ORIGINAL,transform,CACHE

@dataclasses.dataclass
class Meta:
    slot_mapping:object;context_lens:object;block_tables:object;request_block_tables:object
    tree_attention_mask:object;actual_seq_lengths_q:object;sequence_lens:object
    tree_attention:bool=True;use_fused_infer_attention:bool=True


def frame(budgets,variant,device):
    sizes=[]
    for budget in budgets:sizes.extend([5,1]if budget==5 else[budget+1])
    real=sum(sizes);remain=80-real
    while remain:sizes.append(min(5,remain));remain-=min(5,remain)
    ends=[];offset=0
    for q in sizes:offset+=q;ends.append(offset)
    groups=len(sizes);old_capacity=10 if 5 in budgets else 9
    mask=torch.ones((groups,1,old_capacity,4096),dtype=torch.bool,device='cpu')
    for g,qcount in enumerate(sizes):
        for q in range(qcount):mask[g,0,q,:min(2555,1505+g*7+q+variant)]=False
    tables=(torch.arange(groups*32,dtype=torch.int32,device='cpu').reshape(groups,32)+variant*5)%526
    tables[:,20:]=-1
    meta=Meta((torch.arange(80,dtype=torch.long,device='cpu')+variant*1000).to(device),
        torch.arange(80,dtype=torch.int32,device='cpu')+1505+variant,
        tables.to(device),tables.to(device),mask.to(device),tuple(ends),(2555,)*groups)
    return(torch.arange(80,dtype=torch.long,device='cpu').to(device)+variant*17,
           torch.arange(80,dtype=torch.long,device='cpu').to(device)+variant*31,
           meta,real,526),dict(mask_capacity=11 if 5 in budgets else 10)


def compare(actual,expected):
    for a,b in zip(actual[:2]+actual[3:4],expected[:2]+expected[3:4],strict=True):
        assert torch.equal(a.cpu(),b.cpu())
    for field in dataclasses.fields(Meta):
        a,b=getattr(actual[2],field.name),getattr(expected[2],field.name)
        assert torch.equal(a.cpu(),b.cpu())if isinstance(a,torch.Tensor)else a==b
    assert actual[4]==expected[4]


def main():
    output=Path(sys.argv[1]);assert not output.exists()
    torch.set_num_threads(1);torch.npu.set_device(1);torch.npu.config.allow_internal_format=True
    device=torch.device('npu:1')
    spec=importlib.util.spec_from_file_location('_original243_for_NPU_metadata_only',ORIGINAL)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    rows=[]
    for name,budgets in [('N1-B5',[5]),('N8-B5',[5]*8),('N16-B4',[4]*16),('N4-mixed',[1,2,4,5])]:
        frames=[frame(budgets,v,device)for v in (0,1)]
        for args,kwargs in frames:
            compare(transform(*args,**kwargs),original.transform(*args,**kwargs))
        for _ in range(8):
            for args,kwargs in frames:original.transform(*args,**kwargs);transform(*args,**kwargs)
        torch.npu.synchronize()
        timings={'original':[],'cached':[]};pairs=[]
        for pair in range(40):
            args,kwargs=frames[pair%2];order=('original','cached')if pair%2==0 else('cached','original')
            measured={}
            for key in order:
                fn=original.transform if key=='original'else transform
                torch.npu.synchronize();begin=time.perf_counter()
                result=fn(*args,**kwargs)
                torch.npu.synchronize();elapsed=(time.perf_counter()-begin)*1000
                measured[key]=elapsed;timings[key].append(elapsed)
            pairs.append(dict(order=order,variant=pair%2,milliseconds=measured))
        for args,kwargs in frames:compare(transform(*args,**kwargs),original.transform(*args,**kwargs))
        rows.append(dict(case=name,budgets=budgets,paired_calls=40,all_actual_metadata_values_equal=True,
            median_ms={key:statistics.median(values)for key,values in timings.items()},
            mean_ms={key:statistics.mean(values)for key,values in timings.items()},pairs=pairs))
    torch.npu.synchronize()
    payload=dict(schema='ROOT247.actual-NPU-metadata-only-paired-cost.v1',status='complete',
        physical_device=5,logical_device=1,torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,
        rows=rows,cache_hits=CACHE.hits,cache_misses=CACHE.misses,cache_entries=len(CACHE.entries),
        source_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
        candidate_source_sha256=hashlib.sha256(Path(__file__).with_name('projection247.py').read_bytes()).hexdigest(),
        scope='complete metadata transform plus actual device synchronization; prebuilt incoming live metadata',
        models_loaded=False,model_attention_or_KV_executed=False,Goodput=None,
        neural_qualified=False,production_installed=False,adopted=False)
    with output.open('x')as stream:json.dump(payload,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in payload.items()if k!='rows'}))


if __name__=='__main__':main()
