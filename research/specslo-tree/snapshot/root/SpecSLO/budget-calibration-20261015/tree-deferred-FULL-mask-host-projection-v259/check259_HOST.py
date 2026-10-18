"""All227 exact final physical metadata comparisons, CPU protocol only."""
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import json,sys
from pathlib import Path
from fixture259 import build,equal
P=Path(__file__).resolve().parent

def main():
    env=build();engine=env['engine'];host=env['host'];pad=env['pad']
    from oracle255 import cases,prepare
    from branch_tail255 import State
    from projection255 import transform
    from deferred259 import host_factory,branch_prepare,projected
    new,proof=host_factory(host);old=host.make_host_tree_attention_metadata
    state=engine._branch_tail_state;old_prepare=lambda *a,**kw:State.prepare(state,*a,**kw)
    new_prepare=branch_prepare(state);new_transform=projected()
    zero=engine.cache_allocation.block_tables[526][0];assert zero>=0
    count=0
    for case in cases():
        prefixes=[[11]*1505,[12]*2555]+[[13+i]*127 for i in range(22)]
        plans,rows=prepare(case,prefixes);n=case['N'];roots=list(range(41,41+n))
        pp,rr,dd,ss,real=pad(engine,plans,roots,rows,list(range(n)),[],[])
        physical=[p for plan in pp for p in plan.cache_positions.tolist()]
        owners=[owner for owner,plan in zip(ss,pp,strict=True)for _ in range(plan.candidate_budget+1)]
        engine._prepare_attention_metadata(owners,physical,True)
        kwargs=dict(sequence_ids=ss,allow_causal_fast_path=False);q=case['query_count']
        old_raw=old(engine,pp,rr,dd,**kwargs)
        a=old_prepare(pp,ss,*old_raw);a=transform(*a,q,zero,extent=2560,mask_capacity=11 if state.enabled else 10)
        record=dict(state.last_record)
        new_raw=new(engine,pp,rr,dd,**kwargs);assert new_raw[2].tree_attention_mask.device.type=='cpu'
        b=new_prepare(pp,ss,*new_raw);b=new_transform(*b,q,zero,extent=2560,mask_capacity=11 if state.enabled else 10)
        assert record==state.last_record;equal(a,b);count+=1
    assert count==227 and 'torch_npu'not in sys.modules
    row=dict(status='HOST_all227_final_physical_metadata_exact',cases=count,
        values_dtypes_strides_aliases_mapping_and_copy_geometry_equal=True,
        original_HOST_and_branch_and_projection_full_AST_inverses=True,
        proof=proof,dynamic_masks_fresh=True,actual_NPU_used=False,neural_qualified=False,performance_qualified=False)
    with(P/'HOST259_RESULT.json').open('x')as f:json.dump(row,f,indent=2);f.write('\n')
    print(json.dumps(row))
if __name__=='__main__':main()
