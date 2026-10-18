"""Target-only independent math; no original full startup or serving run."""
def run(engine,native,stage,output,fixture):
    from install233 import activate
    activation=stage('canonical-math/activate-diagnostic-only-backend',lambda:activate(engine,native))
    from bank231 import build
    from neural_oracle231 import run as math
    bank=build(engine,native,stage,fixture['prefixes'])
    from runtime233 import save
    save(output,f'rank{engine.rank}-independent-bank.json',dict(rank=engine.rank,bank=bank['evidence'],roots=bank['roots']))
    result=math(engine,native,stage,bank,output=output)
    stage('canonical-math/release-private-neural-bank',lambda:engine._release_cache()if not engine.is_draft else None)
    return dict(schema='ROOT233.bounded-independent-M80-tree-node-math.v1',rank=engine.rank,activation=activation,math=result,bank=bank['evidence'],original_first_roots=bank['roots'],
        original_greedy_sampler_unchanged=True,weights_changed=False,branch_attention_cache_view_modified=True,FP32_tie_policy_installed=False,
        linear_FULL9_and_branch_copy_FULL10=True,real_budgets_B1_to_B5=True,physical_padding_root_only_supported=True,
        whole_workload_AR_qualified=False,original_startup201_44_114_not_run=True,performance=False,formal_Goodput=False,adopted=False)
