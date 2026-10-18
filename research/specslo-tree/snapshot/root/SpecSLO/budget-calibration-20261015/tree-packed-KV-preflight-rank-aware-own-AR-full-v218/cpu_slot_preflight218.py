"""Actual Torch CPU integer/dtype/view control, no Native model or NPU load."""
import datetime,hashlib,json,os,sys,traceback,typing
from pathlib import Path
import types
import slot_preflight218 as subject


def main():
    import torch
    output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=False)
    need=lambda value,message:subject.need(value,message)
    need('torch_npu'not in sys.modules,'CPU-only controlled import; autoload disabled')
    ns=dict(torch=torch,Any=typing.Any,Mapping=typing.Mapping,Sequence=typing.Sequence,
        SpecRhythmPipelineController=types.SimpleNamespace,TreeSpeculationPlan=types.SimpleNamespace,
        _dt_validate_deferred_eager=lambda *a:None)
    before,after=subject.functions(ns);cases=[]
    for dtype in(torch.int32,torch.int64):
        for n in(1,2,8,9,16,24,32):
            for strided in(False,True):
                count=min(4,96//n);budgets=[count]*n;q=sum(b+1 for b in budgets)
                plans=[types.SimpleNamespace(candidate_budget=b,parent_indices=torch.arange(b,device='cpu'))for b in budgets]
                raw=torch.arange(7,7+q*(2 if strided else 1),dtype=dtype,device='cpu')
                if strided:raw=raw[::2]
                engine=types.SimpleNamespace(is_draft=False,device=torch.device('cpu'),model=object(),
                    _tree_cache_capacity=69632,_spec_rhythm_tree_cache_device=lambda d:d)
                control=types.SimpleNamespace(validate_verification=lambda i:types.SimpleNamespace(proposal_id=100+i))
                args=(engine,control,{}, {},list(range(n)),plans,dict(cache_slot_mapping=raw))
                original=before(*args,return_host_mappings=True);current=after(*args,return_host_mappings=True)
                subject.compare_results(original,current)
                need(current[1]=={100+i:raw[i*(count+1):(i+1)*(count+1)].tolist()for i in range(n)},'actual CPU dtype/view exact row values')
                cases.append(dict(dtype=str(dtype),N=n,Q=q,strided=strided,passed=True))
    rejected=[]
    for bad in(-1,69632,2**40):
        raw=torch.tensor([bad,4,5,6,7],dtype=torch.int64,device='cpu')
        plan=types.SimpleNamespace(candidate_budget=4,parent_indices=torch.arange(4,device='cpu'))
        args=(engine,control,{}, {},[0],[plan],dict(cache_slot_mapping=raw))
        errors=[]
        for fn in(before,after):
            try:fn(*args,return_host_mappings=True)
            except RuntimeError as e:errors.append(str(e))
        need(len(errors)==2 and errors[0]==errors[1],'signed64 boundary errors preserved without narrowing')
        rejected.append(dict(value=bad,both_rejected=True,message=errors[0]))
    result=dict(schema='ROOT218.actual-Torch-CPU-mapping-control.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='passed',cases=len(cases),rows=cases,negative_cases=rejected,
        Torch_imported=True,Torch_npu_imported='torch_npu'in sys.modules,NPU_used=False,model_constructed=False,
        original_and_optimized_same_values_and_pending_views=True,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in(subject.NATIVE,subject.CANONICAL,Path(subject.__file__),Path(__file__))})
    (output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in('rows','source_sha256')}))


if __name__=='__main__':main()
