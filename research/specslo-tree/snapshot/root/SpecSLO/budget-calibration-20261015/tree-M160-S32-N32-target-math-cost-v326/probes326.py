"""Same-instance real M160 tree and independently prefilled path math."""
def run(engine,native,stage,output,fixture):
    from pathlib import Path
    import json
    from static_state326 import install
    from bank326 import build
    from oracle326 import run as qualify,CASE_COUNT
    if CASE_COUNT!=532:raise RuntimeError('explicit declared532 case set')
    from raw_controls326 import run as raw_controls
    raw_proof=stage('ROOT326/fresh165-raw-operator-graph-controls',lambda:raw_controls(engine))
    activation=stage('ROOT326/activate-new-M160-backend',lambda:install(engine,native))
    bank=build(engine,native,stage,fixture['prefixes'])
    folder=Path(output)/'independent-node-math'
    stage('ROOT326/create-math-output',lambda:folder.mkdir(exist_ok=False)if engine.rank==0 else None)
    math=qualify(engine,native,stage,bank,output=folder)
    math_calls=list(engine._branch_tail_state.static_graph_calls)
    from audit_static_calls326 import audit_calls
    from oracle326 import cases
    dispatch_audit=stage('ROOT326/actual-cold-dispatch-audit',lambda:audit_calls(cases(),math_calls)if not engine.is_draft else None)
    def checkpoint():
        from runtime317 import save
        save(output,f'rank{engine.rank}-completed-math-before-cost.json',dict(rank=engine.rank,cases=532,math=math,bank=bank['evidence'],actual_static_graph_calls=math_calls,dispatch_audit=dispatch_audit,performance=False))
    stage('ROOT326/retain-math-before-cost',checkpoint)
    def strict_before_costs():
        if engine.is_draft:return None
        if not all(c['hidden_bitwise_equal']and c['KV_bitwise_equal']and c['graph_eager_KV_bitwise_equal']for c in math['cases']):
            raise RuntimeError('ROOT326 strict all532 bitwise gate before cost measurement')
        return True
    stage('ROOT326/all532-target-bitwise-before-costs',strict_before_costs)
    from benchmark326 import run as paired_cost
    costs=paired_cost(engine,native,stage,bank)
    def protected_after_cost():
        if engine.is_draft:return None
        from dual_bank_v146 import prefix_evidence
        for owner,expected in enumerate(bank['evidence']['all_prefix_evidence']):
            if prefix_evidence(engine,{owner:bank['prefixes'][owner%32]})!=expected:
                raise RuntimeError('ROOT326 paired costs changed an independent committed prefix')
        return True
    protected=stage('ROOT326/all96-prefixes-still-exact-after-costs',protected_after_cost)
    stage('ROOT326/release-independent-bank',lambda:engine._release_cache()if not engine.is_draft else None)
    return dict(rank=engine.rank,fresh160_raw_operator=raw_proof,activation=activation,math=math,bank=bank['evidence'],paired_target_cost=costs,dispatch_audit=dispatch_audit,all_prefixes_still_exact_after_costs=protected,
        physical_M=160,physical_query_sizes=[5]*32,
        actual_static_graph_calls=math_calls,
        actual_benchmark_static_graph_calls=engine._branch_tail_state.static_graph_calls[len(math_calls):],
        actual_model_cases=532,whole_workload_AR_qualified=False,performance=False,adopted=False)
