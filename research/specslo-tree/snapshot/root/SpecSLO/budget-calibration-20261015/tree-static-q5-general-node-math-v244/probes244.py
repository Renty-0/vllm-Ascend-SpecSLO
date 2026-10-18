"""Explicit bounded static geometry; no production or workload qualification."""
def run(engine,native,stage,output,fixture):
    from static_state244 import install
    activation=stage('static244/activate-bounded-adapter',lambda:install(engine,native))
    from bank244 import build
    bank=build(engine,native,stage,fixture['prefixes'])
    from runtime244 import save
    save(output,f'rank{engine.rank}-independent-bank.json',dict(rank=engine.rank,bank=bank['evidence'],roots=bank['roots']))
    from oracle244 import run as math
    result=math(engine,native,stage,bank,output=output)
    def evidence():
        state=engine._branch_tail_state
        return list(state.static_graph_calls)
    calls=stage('static244/retain-actual-physical-graph-evidence',evidence)
    stage('static244/release-owned-cache',lambda:engine._release_cache()if not engine.is_draft else None)
    return dict(schema='ROOT244.bounded-static-q5-independent-node-math.v1',rank=engine.rank,activation=activation,
                math=result,bank=bank['evidence'],actual_static_graph_calls=calls,
                original_greedy_sampler_unchanged=True,weights_changed=False,FP32_tie_policy_installed=False,
                original_startup201_44_114_not_run=True,whole_workload_AR_qualified=False,
                supported_primary_plus_side_groups=16,context_extent=2560,
                performance=False,formal_Goodput=False,adopted=False)
