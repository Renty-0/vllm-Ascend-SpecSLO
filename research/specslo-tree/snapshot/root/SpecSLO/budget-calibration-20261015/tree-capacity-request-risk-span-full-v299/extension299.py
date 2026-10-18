"""Fresh same-instance fixed FULL M120 model and own-history protocol."""
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
    return dict(result, ROOT299_canonical_activation_after_original_component=True,
                original_sampler_unchanged=True, effective_limits_applied_only_to_LIVE_generation=True, actual_static_q5_projection=True, fixed_KV_extent=2560)


def loaded_origins():
    import extension218
    from check299 import ready
    value = ready()
    return dict(extension218.loaded_origins(), ROOT299_explicit_sources={
        path:digest for path,digest in value['files'].items() if Path(path).parent == P})


def qualify_component(engine, native, stage, policy, output):
    import extension218
    original = extension218.qualify_component(engine, native, stage, policy, output)
    if engine.cache_allocation is not None or engine.graph_runner.graph_cache_sealed:
        raise RuntimeError('ROOT299 original component must finish and release before activation')
    from static_state299 import install
    activation = stage('ROOT299/activate-new-private-canonical-backend', lambda:install(engine, native))
    engine._canonical299_phase = 'qualification'
    original_pad = native.NativePearlEngine._pad_spec_rhythm_target_tree_graph

    def guarded_pad(e, *args, **kwargs):
        if e is not engine:
            raise RuntimeError('ROOT299 canonical model instance changed')
        if e._canonical299_phase == 'qualified':
            from scope299 import current
            current(e)
        elif e._canonical299_phase != 'qualification':
            raise RuntimeError('ROOT299 neither local qualification nor issued current scope')
        return original_pad(e, *args, **kwargs)

    native.NativePearlEngine._pad_spec_rhythm_target_tree_graph = guarded_pad
    engine._canonical299_pad = guarded_pad
    engine._canonical299_actual_calls = {}
    original_forward = native.NativePearlEngine.target_tree_forward

    def observed_forward(e, *args, **kwargs):
        before_calls=len(e._branch_tail_state.static_graph_calls)
        result = original_forward(e, *args, **kwargs)
        if e._canonical299_phase == 'qualified' and not e.is_draft:
            generation = getattr(e, '_dynamic_tree_generation', None)
            if type(generation) is not int:
                raise RuntimeError('ROOT299 real production generation required')
            state = e._branch_tail_state
            if len(state.static_graph_calls)!=before_calls+1:
                raise RuntimeError('ROOT299 every real target must dispatch exactly one physical static graph')
            current=state.static_current
            if current is None:raise RuntimeError('ROOT299 actual static projection required')
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
            if not(1<=record['N']<=24 and record['C']<=96 and record['M']==120
                   and record['logical_queries']==record['N']+record['C']
                   and record['primary_side_groups']==record['N']+record['branches']<=24
                   and record['copied']is bool(record['branches'])
                   and record['mask_capacity']==(11 if record['copied']else 10)
                   and record['attention_segments']==24 and record['fixed_KV_extent']==[2560]*24
                   and record['actual_q_partition']==list(range(5,121,5))):
                raise RuntimeError('ROOT299 actual static geometry violates declared scope')
            e._canonical299_actual_calls.setdefault(str(generation), []).append(record)
        return result

    native.NativePearlEngine.target_tree_forward = observed_forward
    engine._canonical299_forward = observed_forward
    folder = Path(output)/'canonical-independent-node-math'
    stage('ROOT299/create-private-math-output', lambda:folder.mkdir(exist_ok=False)
          if engine.rank == engine.topology.draft_leader_rank else None)
    fixture = json.loads((P/'FIXTURE299.json').read_text())
    from bank299 import build
    bank = build(engine, native, stage, fixture['prefixes'])
    from oracle299 import run
    math = run(engine, native, stage, bank, output=folder)
    stage('ROOT299/release-independent-bank', lambda:engine._release_cache() if not engine.is_draft else None)
    proof = dict(activation=activation, bank=bank['evidence'], math=math,
                 original_sampler_unchanged=True, original_component=original,
                 cases=408, actual_static_graph_calls=list(engine._branch_tail_state.static_graph_calls), production_qualified=False, whole_workload_AR_qualified=False)
    from runtime299 import save
    stage('ROOT299/retain-current-independent-math', lambda:save(folder, f'rank{engine.rank}-proof.json', proof))
    engine._canonical299_actual_proof = proof
    # Fresh independent draft math, then original deferred repair after an
    # accepted/rejected genuine branch. These remain numeric controls only.
    draft_folder=Path(output)/'independent-genuine-draft-math'
    stage('ROOT299/create-draft-math-folder',lambda:draft_folder.mkdir(exist_ok=False)if engine.rank==0 else None)
    from neural277 import run as draft_math
    draft_proof=draft_math(engine,native,stage,str(draft_folder),fixture)
    follow_folder=Path(output)/'independent-postbranch-draft-math'
    stage('ROOT299/create-followup-folder',lambda:follow_folder.mkdir(exist_ok=False)if engine.rank==0 else None)
    from followup281 import run as followup
    follow_proof=followup(engine,native,stage,str(follow_folder),fixture,save=save)
    engine._ROOT299_draft_numeric_proof=dict(primary_alternate45=draft_proof,postbranch20=follow_proof)
    stage('ROOT299/retain-draft-numeric-proof',lambda:save(Path(output),f'rank{engine.rank}-genuine-draft-proof.json',engine._ROOT299_draft_numeric_proof))
    from live299 import install as install_live
    stage('ROOT299/bind-qualified-future-LIVE-callbacks',lambda:install_live(engine,native,policy))
    # Keep original component object identities and audits intact.
    return original


def cache_snapshot():
    import extension218
    return extension218.cache_snapshot()


def postfull_AR(engine, native, stage, windows, output):
    from reference299 import run
    return run(engine, native, stage, windows, output)


def snapshot(engine):
    return dict(actual_calls_by_generation=engine._canonical299_actual_calls,
                effective_requests=24, effective_candidates=96, physical_queries=120,
                configured_requests=32, configured_candidates=96, physical_primary_side_groups=24, fixed_KV_extent=2560,
                legacy218_upgrade_saved_physical_tokens_counter_is_counterfactual_only=True,
                checked_raw_byte_KV_copy_installed=not engine.is_draft,
                actual_checked_branch_layouts=getattr(engine._branch_tail_state,'copy270_layout_checks',0),
                actual_warmed_row_widths=([engine._branch_tail_state.prefix296_copies.width] if not engine.is_draft else []),
                actual_genuine_draft_counts=getattr(engine,'_ROOT299_draft_counts',{}),
                actual_request_gain_observations=getattr(engine,'_ROOT299_request_gain_observations',{}),
                actual_paired_gain=__import__('risk299').snapshot(engine._ROOT299_policy),
                draft_numeric_proof=getattr(engine,'_ROOT299_draft_numeric_proof',None))
