"""Compose four numerical contexts, with a strictly uninstalled numeric observer."""
def run(engine,native,stage,output,fixture):
    from fresh226 import run as run_fresh
    from history226 import run as run_history
    fresh=run_fresh(engine,native,stage,output,fixture['fresh'])
    from runtime226 import save
    save(output,f'rank{engine.rank}-fresh-complete.json',fresh)
    history=run_history(engine,native,stage,output,fixture['history'])
    if len(fresh['cases'])!=2 or len(history['cases'])!=2:raise RuntimeError('four exact diagnostic contexts required')
    return dict(schema='ROOT226.canonical-FULL-attention-with-uninstalled-tie-refinement.v1',rank=engine.rank,
        cases=[dict(kind='fresh',**c)for c in fresh['cases']]+[dict(kind='history',**c)for c in history['cases']],
        observer_installed=False,model_weights_unchanged=True,original_greedy_sampler_unchanged=True,decode_attention_metadata_changed=True,
        observer_rule='FP32 scalar sums over complete local BF16 maximum ties up to4; original global minimum-ID winner if any winning shard overflows',
        no_performance_claim=True,isolated_head_and_dynamic_body_cost_only=True,independent_AR_qualified=False,
        prefill201_and_production_qualification_not_run=True,formal_Goodput=False,adopted=False)
