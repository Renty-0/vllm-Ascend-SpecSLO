"""Actual Native CPU protocol + actual full333 observer, all532 declarations."""
import ast
import copy
import json
from pathlib import Path
import sys
import types

P=Path(__file__).resolve().parent
B=P.parent
import runtime333
# Frozen fixture reconstructs original Native code, not an alternate protocol.
import check_base326_HOST


def main():
    from fixture326_HOST import build
    env=build();engine=env['engine'];native=env['native']
    from padding326 import pad
    from plan326 import make
    from oracle333 import cases
    source=P/'extension333.py'
    node=copy.deepcopy(next(n for n in ast.walk(ast.parse(source.read_text()))
         if isinstance(n,ast.FunctionDef)and n.name=='observed_forward'))
    ns=dict(original_forward=native.NativePearlEngine.target_tree_forward)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source)+'::HOST-observer','exec'),ns)
    forward=ns['observed_forward']
    engine._canonical333_phase='qualified';engine._dynamic_tree_generation=1;engine._canonical333_actual_calls={}
    results=[]
    for case in cases():
        budgets=case['budgets'];n=len(budgets)
        plans=[make(b,127,case['anchor'])for b in budgets]
        roots=list(range(41,41+n));rows=[[103+17*i+29*j for j in range(b)]for i,b in enumerate(budgets)]
        pp,rr,dd,ss,real=pad(engine,plans,roots,rows,list(range(n)),[],[])
        for owner,plan in enumerate(plans):
            length=plan.prefix_len+plan.candidate_budget+1
            engine._prepare_attention_metadata([owner]*length,list(range(length)),True)
        owners=[owner for owner,plan in zip(ss,pp,strict=True)for _ in range(plan.candidate_budget+1)]
        positions=[v for plan in pp for v in plan.cache_positions.tolist()]
        engine._prepare_attention_metadata(owners,positions,True)
        out=forward(engine,pp,rr,dd,sequence_ids=ss,return_logits=False,real_tree_count=real)
        state=engine._branch_tail_state;meta=state.static_current['physical'][2];q=n+sum(budgets)
        assert out['query_count']==160 and out['logical_query_count']==q and out['graph_padding_query_count']==160-q
        assert out['num_draft_tokens']==budgets and out['tree_count']==n
        assert out['target_query_token_ids'].tolist()==[v for root,row in zip(roots,rows,strict=True)for v in[root]+row]
        assert out['cache_slot_mapping'].numel()==out['cache_slot_mapping'].unique().numel()==q
        assert meta.actual_seq_lengths_q==tuple(range(5,161,5))and meta.sequence_lens==(2560,)*32
        assert tuple(meta.tree_attention_mask.shape)==(32,1,11 if any(b>4 for b in budgets)else 10,4096)
        results.append(dict(name=case['name'],N=n,B=budgets,real_queries=q,physical_queries=160))
    assert len(results)==len(engine._canonical333_actual_calls['1'])==532
    assert any(r['N']==32 and sum(r['B'])==128 and r['real_queries']==160 for r in results)
    # Actual new LIVE program must send32 into the original admission delegate.
    from source333_HOST import source as live_source
    from generation333 import patch
    tree=ast.parse(patch(live_source()))
    caps=[ast.literal_eval(n.value)for n in ast.walk(tree)if isinstance(n,ast.keyword)and n.arg=='max_target_requests']
    assert caps==[32]
    roofs=[ast.unparse(n.value)for n in ast.walk(tree)if isinstance(n,ast.keyword)and n.arg=='verification_budget'and ast.unparse(n.value)=='self._ROOT333_policy.candidate_roof']
    assert len(roofs)==1
    projected,changes=runtime333.project_source((B/'tree-causal80-small-cohort-role-aware-audit-full-v209/runtime209.py').read_text())
    assert len(changes)==15 and 'from config333 import make_config'in projected
    assert 'torch_npu'not in sys.modules
    report=dict(status='HOST_actual_Native532_full333_observer_and_LIVE_target32_passed',
        case_count=532,complete_runtime_projection_inverse=True,
        all_real_tokens_and_slots_preserved=True,physical_FIA_M160_S32=True,
        actual_compiled_LIVE_target32=True,actual_new_constructor_module_selected=True,
        model='CPU arithmetic protocol double; no attention or NPU math',
        NPU_used=False,neural_qualified=False,Goodput=None,cases=results)
    with(P/'HOST_FULL333_PROTOCOL_RESULT.json').open('x')as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in report.items()if k!='cases'}))

if __name__=='__main__':main()
