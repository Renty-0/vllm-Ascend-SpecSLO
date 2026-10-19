"""One negative diagnostic pair; no equivalence or performance permit."""
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
    from trace379 import Trace
    target=not engine.is_draft
    activation=stage('ROOT379/install-frozen-adaptive-prototype',lambda:dict(physical=install(engine,native),KV=bind(engine)))
    def flushed():
        if target and any(m not in engine.model._mc2_chained_row_counts or m in engine.model._mc2_deferred_row_counts for m in (80,120)):
            raise RuntimeError('ROOT379 per-route flushed MC2 required for synchronized observers')
        return True
    stage('ROOT379/check-safe-per-route-flush',flushed)
    tracer=stage('ROOT379/register-value-preserving-observers',lambda:Trace(engine)if target else None)
    case=next(c for c in cases()if c['name']=='N16-all4-anchor0')
    modes=[]
    try:
        for label,enabled in [('fixed120',False),('adaptive80',True)]:
            lane=lambda name,fn:stage('ROOT379/'+label+'/'+name,fn)
            engine._branch_tail_state.adaptive377_enabled=enabled
            bank=build(engine,native,lane,fixture['prefixes'])
            def observe():
                if not target:return None
                state=engine._branch_tail_state
                plans,rows=prepare(case,bank['prefixes']);roots=bank['roots'][:16]
                pp,rr,dd,ss,real=engine._pad_spec_rhythm_target_tree_graph(plans,roots,rows,list(range(16)),[],[])
                result=engine.target_tree_forward(pp,rr,dd,sequence_ids=ss,return_logits=False,real_tree_count=real)
                graph_heads=result['target_query_token_ids'].cpu().tolist()
                before=query_KV(engine,result['cache_slot_mapping'].long())
                tracer.lane=label
                try:hidden=state.eager_last()
                finally:tracer.lane=None
                heads=engine.model.compute_greedy_tokens(hidden[:80],engine.draft_vocab_size).cpu().tolist()
                after=query_KV(engine,result['cache_slot_mapping'].long())
                kv_equal=all(torch.equal(a,b)for a,b in zip(before,after,strict=True))
                x,p,metadata,_,_=state.static_current['physical']
                return dict(mode=label,M=int(x.numel()),S=len(metadata.sequence_lens),
                    physical_tokens_first80=x[:80].cpu().tolist(),physical_positions_first80=p[:80].cpu().tolist(),
                    graph_heads=graph_heads,eager_heads=heads,graph_eager_greedy_equal=graph_heads==heads,
                    graph_eager_all128_KV_bitwise_equal=kv_equal,
                    actual_uniform_KV=list(metadata.sequence_lens),bank=bank['evidence'],
                    model_metrics=engine.graph_metrics())
            modes.append(lane('capture-eager-real-intermediates',observe))
            def protected():
                if not target:return None
                for owner,expected in enumerate(bank['evidence']['all_prefix_evidence']):
                    if prefix_evidence(engine,{owner:bank['prefixes'][owner%24]})!=expected:
                        raise RuntimeError('ROOT379 independent committed prefixes must remain exact')
                return True
            intact=lane('all72-prefixes-after-diagnostic',protected)
            if target:modes[-1]['all72_prefixes_after_diagnostic']=intact
            lane('release-independent-cache-bank',lambda:engine._release_cache()if target else None)
        observation=stage('ROOT379/compare-real-aligned-intermediates',lambda:tracer.compare()if target else None)
        result=dict(rank=engine.rank,status='observations_retained_not_neural_qualification',
            activation=activation,modes=modes,observation=observation,diagnostic_pairs=1,
            independent_prefills=144 if target else 0,raw_tensors_saved_to_disk=False,
            models_qualified=False,real_draft_generation=False,full463=False,performance=False,
            Raw=None,Goodput=None,adopted=False)
        stage('ROOT379/save-independent-observations',lambda:save(Path(output)/f'rank{engine.rank}-observation379.json',result))
        return result
    finally:
        if tracer is not None:tracer.close()
