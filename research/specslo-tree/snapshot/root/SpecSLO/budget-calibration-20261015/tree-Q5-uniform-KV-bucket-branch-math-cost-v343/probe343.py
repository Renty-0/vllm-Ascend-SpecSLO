"""Six maximum-extent qualifications,408mixed+90short original node cases."""
import json
from pathlib import Path
import traceback


def save(path,value):
    with Path(path).open('x')as stream:
        json.dump(value,stream,indent=2);stream.write('\n')


def run(engine,native,stage,output,fixture):
    failures=0
    def boundary(label,fn):
        def execute():
            nonlocal failures
            try:return fn()
            except BaseException:
                failures+=1
                save(Path(output)/f'rank{engine.rank}-local-stage-failure343-{failures}.json',
                    dict(rank=engine.rank,label=label,error=traceback.format_exc(),performance=False))
                raise
        return stage(label,execute)
    from bucket343 import install
    from bank251 import build
    from domain343 import cases,prelude,short_cases,short_prefixes,mid_prefixes,TOTAL_CASES
    from oracle343 import run_selected
    from dispatch343 import audit_calls
    from benchmark343 import run as paired_cost
    from dual_bank_v146 import prefix_evidence
    target=not engine.is_draft
    activation=boundary('ROOT343/install-original311-plus-uniform-bucket',lambda:install(engine,native))
    reports=[]
    for name,prefixes,selected in [('mixed',fixture['prefixes'],cases()),
            ('mid-residue-preserved',mid_prefixes(fixture['prefixes']),short_cases()),
            ('short-residue-preserved',short_prefixes(fixture['prefixes']),short_cases())]:
        lane=lambda label,fn:boundary('ROOT343/'+name+'/'+label,fn)
        folder=Path(output)/name
        lane('create-output',lambda:folder.mkdir(exist_ok=False)if engine.rank==0 else None)
        bank=build(engine,native,lane,prefixes)
        lane('retain-independent-bank',lambda:save(folder/f'rank{engine.rank}-bank-before-math.json',
            dict(rank=engine.rank,evidence=bank['evidence'],teacher_prefix_lengths=[len(p)for p in prefixes],
                 benchmark_workload_modified=False,performance=False)))
        maximum=None
        if name=='mixed':
            if target:engine._branch_tail_state.uniform_bucket343_enabled=False
            baseline_folder=folder/'maximum-extent-qualification'
            lane('maximum/create-output',lambda:baseline_folder.mkdir(exist_ok=False)if engine.rank==0 else None)
            start=len(engine._branch_tail_state.static_graph_calls)
            maximum=run_selected(engine,native,lambda label,fn:lane('maximum/'+label,fn),bank,
                prelude(),baseline_folder)
            calls=list(engine._branch_tail_state.static_graph_calls[start:])
            maximum_dispatch=lane('maximum/actual-dispatch',lambda:audit_calls(prelude(),calls,prefixes,enabled=False)if target else None)
            maximum=dict(math=maximum,actual_calls=calls,dispatch=maximum_dispatch)
        if target:engine._branch_tail_state.uniform_bucket343_enabled=True
        node_folder=folder/'independent-node-math'
        lane('bucket/create-output',lambda:node_folder.mkdir(exist_ok=False)if engine.rank==0 else None)
        before=engine.graph_metrics()
        start=len(engine._branch_tail_state.static_graph_calls)
        math=run_selected(engine,native,lane,bank,selected,node_folder)
        calls=list(engine._branch_tail_state.static_graph_calls[start:])
        dispatch=lane('bucket/actual-dispatch',lambda:audit_calls(selected,calls,prefixes,enabled=True)if target else None)
        lane('retain-math-before-cost',lambda:save(folder/f'rank{engine.rank}-math-before-cost.json',
            dict(rank=engine.rank,math=math,actual_calls=calls,dispatch=dispatch,performance=False)))
        cost_start=len(engine._branch_tail_state.static_graph_calls)
        costs=paired_cost(engine,native,lane,bank)
        full_cost_calls=engine._branch_tail_state.static_graph_calls[cost_start:]
        compact_keys=('N','real_queries','M','attention_segments','actual_seq_lengths_q','sequence_lens',
            'mask_shape','copy_enabled','copy_grid','execution_mode','used_aclgraph','fallback_reason',
            'actual_capture_key','primary_side_groups','uniform_KV_bucket343')
        compact_cost_calls=[{key:call[key] for key in compact_keys}for call in full_cost_calls]
        import hashlib
        cost_call_digest=hashlib.sha256(json.dumps(full_cost_calls,sort_keys=True).encode()).hexdigest()
        del engine._branch_tail_state.static_graph_calls[cost_start:]
        def protected():
            if not target:return None
            for owner,expected in enumerate(bank['evidence']['all_prefix_evidence']):
                if prefix_evidence(engine,{owner:bank['prefixes'][owner%24]})!=expected:
                    raise RuntimeError('ROOT343 all72 independent prefixes must remain exact after costs')
            return True
        intact=lane('all72-prefixes-after-cost',protected)
        after=engine.graph_metrics()
        reports.append(dict(name=name,prefix_lengths=[len(p)for p in prefixes],bank=bank['evidence'],
            maximum_extent_qualification=maximum,math=math,actual_calls=calls,dispatch=dispatch,
            paired_target_cost=costs,actual_cost_calls=compact_cost_calls,
            reconstructable_full_cost_call_SHA=cost_call_digest,all72_prefixes_after_cost=intact,
            model_metrics_before=before,model_metrics_after=after))
        lane('release-bank',lambda:engine._release_cache()if target else None)
    return dict(rank=engine.rank,activation=activation,banks=reports,
        actual_model_cases=TOTAL_CASES,independent_prefills=216 if target else 0,
        physical_M=120,groups=24,original_reference_FULL2560=True,
        real_draft_generation=False,whole_workload_AR_qualified=False,
        performance=False,adopted=False,Goodput=None)
