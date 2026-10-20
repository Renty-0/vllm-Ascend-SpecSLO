"""Actual target observer callback on CPU metadata carriers, no model execution."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
P=Path(__file__).parent;B=P.parent
sys.path.insert(0,str(B/'tree-parent4-best376-target-binding-host-v390'))
from check390_HOST import carrier
import metadata390,target391
import torch
from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan,pack_selected_tree_plan


def main():
    engine=carrier();engine._parent4_target391_binding={}
    n=16
    plans=[pack_selected_tree_plan(cached_cpu_tree_speculation_plan(2,4,128 if i<n else 0,
        4096,candidate_budget=8),range(4))for i in range(24)]
    roots=list(range(100,124));rows=[[i+j+300 for j in range(4)]for i in range(24)]
    frame=(plans,roots,rows,list(range(24)),n)
    mode=dict(corrupt=None)
    engine.graph_runner=SimpleNamespace(run_target_greedy=lambda *a:(torch.arange(120),))
    def forward(plans,roots,rows,ids,*,return_logits,real_tree_count):
        actual=(plans,roots,rows,ids,real_tree_count)
        x,pos,meta=metadata390.physical(engine,actual)
        if mode['corrupt']=='context_lens':meta=replace(meta,context_lens=meta.context_lens+1)
        elif mode['corrupt']=='sequence_lens':meta=replace(meta,sequence_lens=(1024,)*24)
        elif mode['corrupt']=='input':x=x+1
        state=engine._branch_tail_state
        state.static_current=dict(copy_enabled=False,physical=(x,pos,meta,None,
            dict(physical_to_original=list(range(120)))))
        output=engine.graph_runner.run_target_greedy([],[],[],151936)
        return dict(used_aclgraph=True,num_draft_tokens=[4]*real_tree_count,
            target_query_token_ids=output[0][:real_tree_count*5])
    engine.target_tree_forward=forward
    for changed in (False,True):
        output,full=target391.original_call(engine,frame,changed=changed)
        assert torch.equal(full,torch.arange(120))and len(output['target_query_token_ids'])==80
    assert engine._parent4_target391_binding['actual_metadata_matches']==2
    for field in ('context_lens','sequence_lens','input'):
        mode['corrupt']=field
        try:target391.original_call(engine,frame)
        except RuntimeError as error:assert 'all actual Native physical metadata differs'in str(error)
        else:raise AssertionError('controlled physical mismatch must reject')
    assert engine._parent4_target391_binding['actual_metadata_matches']==2
    assert not torch.npu.is_initialized()
    result=dict(status='PASS_HOST391_REAL_OBSERVER_POSITIVE_AND_CONTROLLED_METADATA_NEGATIVES',
        NPU_used=False,model_forward_called=False,CPU_carriers_only=True,
        actual_observer_function_called=True,real_packer_and_projection=True,
        changed_input_checked=True,negative_fields=['context_lens','sequence_lens','input'],
        numerical_qualification=False,production_permission=False,Raw=None,Goodput=None)
    (P/'HOST391_PHYSICAL_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
