"""Fresh same-instance fixed FULL M160 model and own-history protocol."""
from pathlib import Path
import json

P = Path(__file__).resolve().parent


def prepare_install(output):
    import extension218
    return extension218.prepare_install(output)


def install_entry(entry, native):
    import host_target_metadata as host
    from root_padding_forward231 import host_installer
    original = host.install
    host.install = host_installer(original)
    try:
        return entry['install'](fast=True, host=True, transport=True, replay=True, repair=True, prefill=True)
    finally:
        host.install = original


def after_install(native, graph, live):
    import extension218
    result = extension218.after_install(native, graph, live)
    return dict(result, ROOT333_canonical_activation_after_original_component=True,
                original_sampler_unchanged=True, effective_limits_applied_only_to_LIVE_generation=True, actual_static_q5_projection=True, fixed_KV_extent=2560)


def loaded_origins():
    import extension218
    from check333 import ready
    value = ready()
    return dict(extension218.loaded_origins(), ROOT333_explicit_sources={
        path:digest for path,digest in value['files'].items() if Path(path).parent == P})


def activate_capacity(engine,native,policy,original_proof):
    if (getattr(engine,'_v209_qualified_component',None)is not original_proof
            or engine.cache_allocation is not None or engine.graph_runner.graph_cache_sealed
            or engine.config.spec_rhythm_verification_budget!=96):
        raise RuntimeError('ROOT333 original controls and unchanged constructor required before new capacity')
    from capacity329 import bind
    proof=bind(policy,native)
    proof.update(original_constructor_C96_retained=True,activation_after_original_component=True)
    policy._ROOT333_actual_capacity=proof
    return proof


def qualify_component(engine, native, stage, policy, output):
    import extension218
    original = extension218.qualify_component(engine, native, stage, policy, output)
    if engine.cache_allocation is not None or engine.graph_runner.graph_cache_sealed:
        raise RuntimeError('ROOT333 original component must finish and release before activation')
    stage('ROOT333/bind-C128-only-after-original-controls',lambda:activate_capacity(engine,native,policy,original))
    from raw_controls326 import run as raw_controls
    raw_proof = stage('ROOT333/fresh-raw160-controls', lambda:raw_controls(engine))
    from static_state333 import install
    activation = stage('ROOT333/activate-new-private-canonical-backend', lambda:install(engine, native))
    engine._canonical333_phase = 'qualification'
    original_pad = native.NativePearlEngine._pad_spec_rhythm_target_tree_graph

    def guarded_pad(e, *args, **kwargs):
        if e is not engine:
            raise RuntimeError('ROOT333 canonical model instance changed')
        if e._canonical333_phase == 'qualified':
            from scope333 import current
            current(e)
        elif e._canonical333_phase != 'qualification':
            raise RuntimeError('ROOT333 neither local qualification nor issued current scope')
        return original_pad(e, *args, **kwargs)

    native.NativePearlEngine._pad_spec_rhythm_target_tree_graph = guarded_pad
    engine._canonical333_pad = guarded_pad
    engine._canonical333_actual_calls = {}
    original_forward = native.NativePearlEngine.target_tree_forward

    def observed_forward(e, *args, **kwargs):
        before_calls=len(e._branch_tail_state.static_graph_calls)
        result = original_forward(e, *args, **kwargs)
        if e._canonical333_phase == 'qualified' and not e.is_draft:
            generation = getattr(e, '_dynamic_tree_generation', None)
            if type(generation) is not int:
                raise RuntimeError('ROOT333 real production generation required')
            state = e._branch_tail_state
            if len(state.static_graph_calls)!=before_calls+1:
                raise RuntimeError('ROOT333 every real target must dispatch exactly one physical static graph')
            current=state.static_current
            if current is None:raise RuntimeError('ROOT333 actual static projection required')
            _,_,meta,inverse,mapping=current['physical']
            budgets=result['num_draft_tokens']
            record=dict(N=len(budgets),C=sum(budgets),M=result['query_count'],
                        logical_queries=result['logical_query_count'],branches=sum(b>4 for b in budgets),
                        attention_segments=len(meta.actual_seq_lengths_q),mask_capacity=int(meta.tree_attention_mask.shape[2]),
                        copied=current['copy_enabled'],actual_copy_slots=state.last_record.get('real_copy_slots',0),
                        fixed_KV_extent=list(meta.sequence_lens),actual_q_partition=list(meta.actual_seq_lengths_q),
                        primary_side_groups=mapping['supported_primary_side_groups'],
                        physical_to_original=mapping['physical_to_original'],original_to_physical=mapping['original_to_physical'],
                        copy_grid=getattr(state,'copy280_bucket',0),actual_capture_key=getattr(state,'copy280_last_dispatch_key',None))
            if not(1<=record['N']<=32 and record['C']<=128 and record['M']==160
                   and record['logical_queries']==record['N']+record['C']
                   and record['primary_side_groups']==record['N']+record['branches']<=32
                   and record['copied']is bool(record['branches'])
                   and record['mask_capacity']==(11 if record['copied']else 10)
                   and record['attention_segments']==32 and record['fixed_KV_extent']==[2560]*32
                   and record['actual_q_partition']==list(range(5,161,5))):
                raise RuntimeError('ROOT333 actual static geometry violates declared scope')
            e._canonical333_actual_calls.setdefault(str(generation), []).append(record)
        return result

    native.NativePearlEngine.target_tree_forward = observed_forward
    engine._canonical333_forward = observed_forward
    folder = Path(output)/'canonical-independent-node-math'
    stage('ROOT333/create-private-math-output', lambda:folder.mkdir(exist_ok=False)
          if engine.rank == engine.topology.draft_leader_rank else None)
    fixture = json.loads((P/'FIXTURE333.json').read_text())
    from bank333 import build
    bank = build(engine, native, stage, fixture['prefixes'])
    from oracle333 import run
    math = run(engine, native, stage, bank, output=folder)
    def strict_math():
        if engine.is_draft:return None
        if not all(c['hidden_bitwise_equal'] and c['KV_bitwise_equal'] and c['exact_greedy']
                   and c['graph_eager_KV_bitwise_equal'] for c in math['cases']):
            raise RuntimeError('ROOT333 all532 fresh target cases must be bitwise equal')
        return True
    stage('ROOT333/strict-all532-before-LIVE-scope',strict_math)
    math_calls=list(engine._branch_tail_state.static_graph_calls)
    from benchmark326 import run as calibrate_cost
    costs=calibrate_cost(engine,native,stage,bank)
    cost_directory=Path(output)/'paired-current-target-cost333'
    stage('ROOT333/create-current-cost-output',lambda:cost_directory.mkdir(exist_ok=False)if engine.rank==0 else None)
    from runtime333 import save
    from check333 import sha
    freeze_sha=sha(P/'FREEZE333.json')
    def retain_cost():
        target=not engine.is_draft
        save(cost_directory,f'rank{engine.rank}-target-cost.json',dict(
            rank=engine.rank,source_freeze_sha256=freeze_sha,run_directory=str(Path(output).resolve()),
            model_identity=id(engine.model),model_dtype=str(engine.model.embed_tokens.weight.dtype),
            fresh_math_cases=532,all_math_bitwise=True if target else None,
            physical_FIA_rows=160,physical_FIA_groups=32,actual_projection_rows=165,
            chained_flush_334=True if target else None,deferred_read_done=False,
            profile_sha256=engine.mc2_static_route_manifest.profile_sha256 if target else None,
            costs=costs))
    stage('ROOT333/retain-current-cost-all4-roles',retain_cost)
    def protected_after_cost():
        if engine.is_draft:return None
        from dual_bank_v146 import prefix_evidence
        for owner,wanted in enumerate(bank['evidence']['all_prefix_evidence']):
            if prefix_evidence(engine,{owner:bank['prefixes'][owner%32]})!=wanted:
                raise RuntimeError('ROOT333 cost calibration changed committed independent prefix')
        return True
    protected=stage('ROOT333/protect-all96-prefixes-after-cost',protected_after_cost)
    from cost333 import bind_current
    calibration=stage('ROOT333/bind-same-instance-target-cost',lambda:bind_current(policy,engine,cost_directory,freeze_sha))
    benchmark_calls=len(engine._branch_tail_state.static_graph_calls)-len(math_calls)
    stage('ROOT333/release-independent-bank', lambda:engine._release_cache() if not engine.is_draft else None)
    proof = dict(fresh160_raw_operator=raw_proof, actual_capacity=policy._ROOT333_actual_capacity, activation=activation, bank=bank['evidence'], math=math,
                 original_sampler_unchanged=True, original_component=original,
                 cases=532, actual_static_graph_calls=math_calls, actual_cold_math_graph_calls_only=True, untimed_cost_graph_call_count=benchmark_calls, paired_target_cost_calibration=calibration, all_prefixes_exact_after_cost=protected, production_qualified=False, whole_workload_AR_qualified=False)
    from runtime333 import save
    stage('ROOT333/retain-current-independent-math', lambda:save(folder, f'rank{engine.rank}-proof.json', proof))
    engine._canonical333_actual_proof = proof
    # Fresh independent draft math, then original deferred repair after an
    # accepted/rejected genuine branch. These remain numeric controls only.
    draft_folder=Path(output)/'independent-genuine-draft-math'
    stage('ROOT333/create-draft-math-folder',lambda:draft_folder.mkdir(exist_ok=False)if engine.rank==0 else None)
    from neural277 import run as draft_math
    draft_proof=draft_math(engine,native,stage,str(draft_folder),fixture)
    follow_folder=Path(output)/'independent-postbranch-draft-math'
    stage('ROOT333/create-followup-folder',lambda:follow_folder.mkdir(exist_ok=False)if engine.rank==0 else None)
    from followup281 import run as followup
    follow_proof=followup(engine,native,stage,str(follow_folder),fixture,save=save)
    engine._ROOT333_draft_numeric_proof=dict(primary_alternate45=draft_proof,postbranch20=follow_proof)
    stage('ROOT333/retain-draft-numeric-proof',lambda:save(Path(output),f'rank{engine.rank}-genuine-draft-proof.json',engine._ROOT333_draft_numeric_proof))
    from live333 import install as install_live
    stage('ROOT333/bind-qualified-future-LIVE-callbacks',lambda:install_live(engine,native,policy))
    # Keep original component object identities and audits intact.
    return original


def cache_snapshot():
    import extension218
    return extension218.cache_snapshot()


def postfull_AR(engine, native, stage, windows, output):
    from reference333 import run
    return run(engine, native, stage, windows, output)


def snapshot(engine):
    return dict(actual_calls_by_generation=engine._canonical333_actual_calls,
                effective_requests=32, effective_candidates=128, physical_queries=160,
                configured_requests=32, configured_candidates=96, physical_primary_side_groups=32, fixed_KV_extent=2560,
                legacy218_upgrade_saved_physical_tokens_counter_is_counterfactual_only=True,
                checked_raw_byte_KV_copy_installed=not engine.is_draft,
                actual_checked_branch_layouts=getattr(engine._branch_tail_state,'copy270_layout_checks',0),
                actual_warmed_row_widths=([engine._branch_tail_state.prefix326_copies.width] if not engine.is_draft else []),
                actual_genuine_draft_counts=getattr(engine,'_ROOT333_draft_counts',{}),
                actual_request_gain_observations=getattr(engine,'_ROOT333_request_gain_observations',{}),
                actual_paired_gain=__import__('risk333').snapshot(engine._ROOT333_policy),
                draft_numeric_proof=getattr(engine,'_ROOT333_draft_numeric_proof',None))
