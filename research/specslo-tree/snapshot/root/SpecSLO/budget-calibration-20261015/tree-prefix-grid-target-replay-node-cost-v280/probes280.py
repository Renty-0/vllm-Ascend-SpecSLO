"""Same-instance real M120 tree and independently prefilled path math."""
def run(engine,native,stage,output,fixture):
    from pathlib import Path
    import json
    from static_state280 import install
    from bank251 import build
    from oracle280 import run as qualify,CASE_COUNT
    if CASE_COUNT!=408:raise RuntimeError('explicit declared227 case set')
    activation=stage('ROOT280/activate-new-M120-backend',lambda:install(engine,native))
    bank=build(engine,native,stage,fixture['prefixes'])
    folder=Path(output)/'independent-node-math'
    stage('ROOT280/create-math-output',lambda:folder.mkdir(exist_ok=False)if engine.rank==0 else None)
    math=qualify(engine,native,stage,bank,output=folder)
    math_calls=list(engine._branch_tail_state.static_graph_calls)
    from audit_static_calls280 import audit_calls
    from oracle280 import cases
    dispatch_audit=stage('ROOT280/actual-cold-dispatch-audit',lambda:audit_calls(cases(),math_calls)if not engine.is_draft else None)
    def checkpoint():
        from runtime280 import save
        save(output,f'rank{engine.rank}-completed-math-before-cost.json',dict(rank=engine.rank,cases=408,math=math,bank=bank['evidence'],actual_static_graph_calls=math_calls,dispatch_audit=dispatch_audit,performance=False))
    stage('ROOT280/retain-math-before-cost',checkpoint)
    from benchmark280 import run as paired_cost
    costs=paired_cost(engine,native,stage,bank)
    def protected_after_cost():
        if engine.is_draft:return None
        from dual_bank_v146 import prefix_evidence
        for owner,expected in enumerate(bank['evidence']['all_prefix_evidence']):
            if prefix_evidence(engine,{owner:bank['prefixes'][owner%24]})!=expected:
                raise RuntimeError('ROOT280 paired costs changed an independent committed prefix')
        return True
    protected=stage('ROOT280/all72-prefixes-still-exact-after-costs',protected_after_cost)
    stage('ROOT280/release-independent-bank',lambda:engine._release_cache()if not engine.is_draft else None)
    return dict(rank=engine.rank,activation=activation,math=math,bank=bank['evidence'],paired_target_cost=costs,dispatch_audit=dispatch_audit,all_prefixes_still_exact_after_costs=protected,
        physical_M=120,physical_query_sizes=[5]*24,
        actual_static_graph_calls=math_calls,
        actual_benchmark_static_graph_calls=engine._branch_tail_state.static_graph_calls[len(math_calls):],
        actual_model_cases=408,whole_workload_AR_qualified=False,performance=False,adopted=False)
