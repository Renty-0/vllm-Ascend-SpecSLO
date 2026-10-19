"""Current model693 cases and balanced80/120 target costs; never LIVE scope."""
import hashlib
import json
from pathlib import Path


def save(path,value):
    with Path(path).open('x') as out:
        json.dump(value,out,indent=2)
        out.write('\n')


def run(engine,native,stage,output,fixture):
    from install377 import install
    from bucket377 import bind
    from bank251 import build
    from oracle377 import run as neural, cases
    from dispatch377 import audit_calls
    from prefixes376 import mid_prefixes,short_prefixes
    from benchmark378 import run as costs
    from dual_bank_v146 import prefix_evidence
    target = not engine.is_draft
    def install_current():
        if target:
            for m in (80,120):
                if m not in engine.model._mc2_chained_row_counts or m in engine.model._mc2_deferred_row_counts:
                    raise RuntimeError('ROOT378 actual unchanged flushed MC2 profiles required for80 and120')
        result = install(engine,native)
        extent = bind(engine)
        engine._branch_tail_state.adaptive377_enabled = True
        return dict(physical=result,KV=extent)
    activation = stage('ROOT378/install-one-copy-state-and-visible-KV-bucket',install_current)
    summaries = []
    for name,prefixes in [('mixed',fixture['prefixes']),('mid',mid_prefixes(fixture['prefixes'])),
                          ('short',short_prefixes(fixture['prefixes']))]:
        lane = lambda label,fn:stage('ROOT378/'+name+'/'+label,fn)
        folder = Path(output)/name
        lane('create-bank-output',lambda:folder.mkdir(exist_ok=False) if engine.rank==0 else None)
        bank = build(engine,native,lane,prefixes)
        state = engine._branch_tail_state
        before = len(state.static_graph_calls)
        math = neural(engine,native,lane,bank,folder)
        calls = list(state.static_graph_calls[before:])
        dispatch = lane('reconstruct-all-actual-dispatches',lambda:audit_calls(cases(),calls,prefixes) if target else None)
        checkpoint = folder/f'rank{engine.rank}-math-before-cost.json'
        lane('retain-complete-math-before-cost',lambda:save(checkpoint,
            dict(rank=engine.rank,bank=bank['evidence'],prefix_lengths=[len(p)for p in prefixes],
                 math=math,actual_calls=calls,dispatch=dispatch,performance=False)))
        before = len(state.static_graph_calls)
        panel = costs(engine,native,lane,bank)
        cost_calls = list(state.static_graph_calls[before:])
        del state.static_graph_calls[before:]
        def protected():
            if not target:return None
            for owner,expected in enumerate(bank['evidence']['all_prefix_evidence']):
                if prefix_evidence(engine,{owner:bank['prefixes'][owner%24]})!=expected:
                    raise RuntimeError('ROOT378 all72 committed prefix KV tensors must remain exact')
            return True
        intact = lane('all72-prefixes-after-balanced-cost',protected)
        cost_path = folder/f'rank{engine.rank}-cost-and-prefix-protection.json'
        lane('retain-paired-cost',lambda:save(cost_path,dict(rank=engine.rank,name=name,
            paired_physical_cost=panel,actual_cost_calls=cost_calls,all72_prefixes_after_cost=intact,
            full463=False,real_draft_generation=False,Raw=None,Goodput=None)))
        summaries.append(dict(name=name,prefix_lengths=[len(p)for p in prefixes],
            math_path=str(checkpoint),math_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            cost_path=str(cost_path),cost_sha256=hashlib.sha256(cost_path.read_bytes()).hexdigest(),cases=231))
        lane('release-independent-bank',lambda:engine._release_cache() if target else None)
    return dict(rank=engine.rank,activation=activation,banks=summaries,actual_model_cases=693,
        independent_prefills=216 if target else 0,physical_M=[80,120],groups=[16,24],Q=5,
        real_budget_domain=[1,2,3,4,5],independent_reference_FULL120=True,
        real_draft_generation=False,whole_workload_AR_qualified=False,performance=False,
        adopted=False,Raw=None,Goodput=None)
