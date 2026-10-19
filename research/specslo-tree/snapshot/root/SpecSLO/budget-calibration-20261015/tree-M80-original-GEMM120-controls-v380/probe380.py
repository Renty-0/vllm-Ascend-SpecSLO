"""Independent banks, matched real inputs, three eager-only GEMM controls."""
import json
from pathlib import Path


def save(path,value):
    with Path(path).open('x')as out:json.dump(value,out,indent=2);out.write('\n')


def run(engine,native,stage,output,fixture):
    import torch
    from install377 import install
    from bucket377 import bind
    from bank251 import build
    from oracle255 import cases,prepare
    from dual_bank_v146 import query_KV,prefix_evidence
    from control380 import Control
    from trace380 import Trace
    target=not engine.is_draft
    activation=stage('ROOT380/install-original-adaptive-prototype',lambda:dict(physical=install(engine,native),KV=bind(engine)))
    def flushed():
        if target and any(m not in engine.model._mc2_chained_row_counts or m in engine.model._mc2_deferred_row_counts for m in (80,120)):
            raise RuntimeError('ROOT380 safe flushed MC2 observation required')
        return True
    stage('ROOT380/check-per-route-flush',flushed)
    tracer=stage('ROOT380/register-observers',lambda:Trace(engine)if target else None)
    case=next(c for c in cases()if c['name']=='N16-all4-anchor0')
    modes=[];reference=None
    try:
        for mode in ['fixed120','qkv120','columns120','columns_and_mc2_120']:
            lane=lambda name,fn:stage('ROOT380/'+mode+'/'+name,fn)
            engine._branch_tail_state.adaptive377_enabled=mode!='fixed120'
            bank=build(engine,native,lane,fixture['prefixes'])
            def observe():
                nonlocal reference
                if not target:return None
                state=engine._branch_tail_state
                plans,rows=prepare(case,bank['prefixes']);roots=bank['roots'][:16]
                pp,rr,dd,ss,real=engine._pad_spec_rhythm_target_tree_graph(plans,roots,rows,list(range(16)),[],[])
                graph=engine.target_tree_forward(pp,rr,dd,sequence_ids=ss,return_logits=False,real_tree_count=real)
                graph_heads=graph['target_query_token_ids'].cpu().tolist()
                if mode!='fixed120':tracer.forget_candidate()
                with Control(engine,'none'if mode=='fixed120'else mode)as control:
                    tracer.lane='fixed120'if mode=='fixed120'else'adaptive80'
                    try:hidden=state.eager_last()
                    finally:tracer.lane=None
                    heads=engine.model.compute_greedy_tokens(hidden[:80],engine.draft_vocab_size).cpu().tolist()
                    kv=query_KV(engine,graph['cache_slot_mapping'].long())
                    if mode=='fixed120':reference=(hidden[:80].detach().clone(),kv,heads)
                    ref_hidden,ref_kv,ref_heads=reference
                    matcher=engine.graph_runner._target_outputs_match
                    hidden_match=matcher((hidden[:80],),(ref_hidden,))
                    KV_equal=[torch.equal(a,b)for a,b in zip(kv,ref_kv,strict=True)]
                    KV_match=matcher(tuple(kv),tuple(ref_kv))
                x,p,metadata,_,_=state.static_current['physical']
                return dict(mode=mode,M=int(x.numel()),S=len(metadata.sequence_lens),
                    physical_tokens_first80=x[:80].cpu().tolist(),physical_positions_first80=p[:80].cpu().tolist(),
                    original_graph_heads=graph_heads,controlled_eager_heads=heads,
                    reference_greedy_equal=heads==ref_heads,
                    greedy_difference_rows=[i for i,(x,y)in enumerate(zip(ref_heads,heads))if x!=y],
                    hidden_original_matcher_passed=hidden_match,
                    hidden_max_abs=float((hidden[:80].float()-ref_hidden.float()).abs().max().item()),
                    KV_all128_bitwise_equal=all(KV_equal),KV_original_matcher_passed=KV_match,
                    control=control.counts,bank=bank['evidence'],
                    observation=None if mode=='fixed120'else tracer.compare())
            modes.append(lane('actual-controlled-eager-observations',observe))
            def protected():
                if not target:return None
                for owner,expected in enumerate(bank['evidence']['all_prefix_evidence']):
                    if prefix_evidence(engine,{owner:bank['prefixes'][owner%24]})!=expected:
                        raise RuntimeError('ROOT380 all committed prefixes protected')
                return True
            intact=lane('all72-committed-prefixes',protected)
            if target:
                modes[-1]['all72_prefixes_protected']=intact
                lane('retain-each-control',lambda:save(Path(output)/f'rank{engine.rank}-{mode}-control380.json',modes[-1]))
            lane('release-independent-bank',lambda:engine._release_cache()if target else None)
        result=dict(rank=engine.rank,status='three_eager_GEMM_controls_observed_NOT_qualification',
            activation=activation,modes=modes,diagnostic_controls=3,independent_prefills=288 if target else 0,
            model_qualified=False,performance=False,full463=False,adopted=False,Raw=None,Goodput=None)
        stage('ROOT380/save-observations',lambda:save(Path(output)/f'rank{engine.rank}-observation380.json',result))
        return result
    finally:
        if tracer is not None:tracer.close()
