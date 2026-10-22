"""Actual CPU metadata/observer with only the real425 state field present."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    import torch
    from check390_HOST import carrier
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan,pack_selected_tree_plan
    import metadata390,metadata428,target428
    engine=carrier()
    engine._branch_tail_state=SimpleNamespace(uniform_KV_bucket425_enabled=True)
    engine._parent4_target391_binding={}
    engine.graph_runner=SimpleNamespace(run_target_greedy=lambda *a:(torch.arange(120),))
    mode=dict(corrupt=None)
    def forward(plans,roots,rows,ids,*,return_logits,real_tree_count):
        x,pos,meta=metadata428.physical(engine,(plans,roots,rows,ids,real_tree_count))
        if mode['corrupt']=='context_lens':meta=replace(meta,context_lens=meta.context_lens+1)
        elif mode['corrupt']=='sequence_lens':meta=replace(meta,sequence_lens=(1024,)*24)
        elif mode['corrupt']=='input':x=x+1
        engine._branch_tail_state.static_current=dict(copy_enabled=False,
            physical=(x,pos,meta,None,dict(physical_to_original=list(range(120)))))
        output=engine.graph_runner.run_target_greedy([],[],[],151936)
        return dict(used_aclgraph=True,num_draft_tokens=[4]*real_tree_count,
                    target_query_token_ids=output[0][:5*real_tree_count])
    engine.target_tree_forward=forward
    cases=0;old_reproduced=False
    for n in range(1,25):
        for prefix in (128,511,1500,2555):
            plans=[pack_selected_tree_plan(cached_cpu_tree_speculation_plan(
                2,4,prefix if i<n else 0,4096,candidate_budget=8),range(4))for i in range(24)]
            roots=list(range(100,124));rows=[[i+j+300 for j in range(4)]for i in range(24)]
            frame=(plans,roots,rows,list(range(24)),n)
            if not old_reproduced:
                assert not hasattr(engine._branch_tail_state,'uniform_KV_bucket376_enabled')
                try:metadata390.physical(engine,frame)
                except AttributeError as error:assert 'uniform_KV_bucket376_enabled'in str(error)
                else:raise AssertionError('old425/390 binding failure not reproduced')
                old_reproduced=True
            for changed in (False,True):
                x,pos,meta=metadata428.physical(engine,frame,changed=changed)
                expected=((prefix+5+511)//512)*512
                assert tuple(meta.sequence_lens)==(expected,)*24
                assert meta.actual_seq_lengths_q==tuple(range(5,121,5))
                assert x[::5].tolist()==[r+int(changed)for r in roots]
                output,full=target428.original_call(engine,frame,changed=changed)
                assert len(output['target_query_token_ids'])==5*n
                assert torch.equal(full,torch.arange(120))
                cases+=1
    assert engine._parent4_target391_binding['actual_metadata_matches']==cases==192
    for field in ('context_lens','sequence_lens','input'):
        mode['corrupt']=field
        try:target428.original_call(engine,frame)
        except RuntimeError as error:assert 'all actual Native physical metadata differs'in str(error)
        else:raise AssertionError('controlled real metadata mismatch accepted')
    assert engine._parent4_target391_binding['actual_metadata_matches']==192
    mode['corrupt']=None
    engine._branch_tail_state.uniform_KV_bucket425_enabled=False
    engine._branch_tail_state.uniform_KV_bucket376_enabled=True
    try:metadata428.physical(engine,frame)
    except RuntimeError as error:assert 'actual best376 target extent policy required'in str(error)
    else:raise AssertionError('old true flag overrode false actual425 state')
    assert not torch.npu.is_initialized()
    value=dict(status='PASS_HOST428_REAL_CURRENT_STATE_METADATA_AND_OBSERVER',
        CPU_cases=cases,old_missing376_field_failure_reproduced=True,
        actual_current425_field_only=True,actual_native_packer_and_static_transform=True,
        original_target391_observer_body_used=True,changed_roots_only_checked=True,
        controlled_metadata_negatives=['context_lens','sequence_lens','input'],
        actual_false_flag_cannot_be_overridden_by_legacy_true=True,
        source=metadata428.descriptor(),NPU_used=False,Native_models_constructed=0,
        NN_math_qualified=False,production_permission=False,Raw=None,Goodput=None)
    p=Path(__file__).resolve().parent
    (p/'HOST428_METADATA_RESULT.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
