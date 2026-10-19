"""Actual checked Native frontend and independent metadata; CPU fixture only."""
import json
from pathlib import Path
import sys

P=Path(__file__).resolve().parent;B=P.parent
sys.path.insert(0,str(B/'tree-Q5-bucket-causal-inner1-target-math-cost-v341'))
sys.path.insert(0,str(B/'tree-Q4-closure-aware-host-builder-adapter-v339'))
from check339_HOST import fixture


def main():
    engine,ignored=fixture()
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    import metadata341
    activation=metadata341.install(engine)
    engine._activate_cache_sequence(32,[31]*97,enable_prefix_caching=False)
    outputs=[]
    for family in ('FULL2560','FULLbucket'):
        selected=2560 if family=='FULL2560' else 512
        engine._shape341=(family,selected)
        ids=[0]+[463+i for i in range(23)]
        plans=[cached_cpu_tree_speculation_plan(1,4,97 if owner==0 else 0,4096,candidate_budget=4)for owner in ids]
        out=engine.target_tree_forward(plans,[41]*24,[[43]*4 for _ in ids],sequence_ids=ids,return_logits=False,real_tree_count=1)
        assert out['query_count']==120 and out['logical_query_count']==5
        assert out['used_tree_attention']is(family!='causal-inner1')
        call=engine._calls341[-1]
        assert call['sequence_lens']==([102]+[5]*23 if family=='causal-inner1' else [selected]*24)
        for future in (0,17):
            for depth in range(5):
                x,pos,meta=metadata341.reference_frame(engine,dict(family=family),owners=[32],positions=[97+depth],tokens=[41],future=future,extent=selected)
                assert x.numel()==120 and meta.tree_attention is(family!='causal-inner1')
                assert len(meta.sequence_lens)==24
        outputs.append(call)
    assert ignored==[] and len(engine._HOST339_account_calls)==2
    assert 'torch_npu'not in sys.modules
    result=dict(status='passed_actual_frontend_and_reference_metadata_CPU',NPU_used=False,
        physical_M=120,groups=24,Native_accounting_preserved=True,
        all20_reference_frames_passed=True,actual_calls=outputs,activation=activation,
        fake_graph_not_NPU_evidence=True)
    (P/'HOST342_METADATA_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
