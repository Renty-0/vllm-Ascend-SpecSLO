"""Original218 qualification first, then same-instance contiguous FULL math."""
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
    return dict(result, ROOT235_canonical_activation_after_original_component=True,
                original_sampler_unchanged=True, effective_limits_applied_only_to_LIVE_generation=True)


def loaded_origins():
    import extension218
    from check235 import ready
    value = ready()
    return dict(extension218.loaded_origins(), ROOT235_explicit_sources={
        path:digest for path,digest in value['files'].items() if Path(path).parent == P})


def qualify_component(engine, native, stage, policy, output):
    import extension218
    original = extension218.qualify_component(engine, native, stage, policy, output)
    if engine.cache_allocation is not None or engine.graph_runner.graph_cache_sealed:
        raise RuntimeError('ROOT235 original component must finish and release before activation')
    from branch_tail231 import install
    activation = stage('ROOT235/activate-new-private-canonical-backend', lambda:install(engine, native))
    engine._canonical235_phase = 'qualification'
    original_pad = native.NativePearlEngine._pad_spec_rhythm_target_tree_graph

    def guarded_pad(e, *args, **kwargs):
        if e is not engine:
            raise RuntimeError('ROOT235 canonical model instance changed')
        if e._canonical235_phase == 'qualified':
            from scope235 import current
            current(e)
        elif e._canonical235_phase != 'qualification':
            raise RuntimeError('ROOT235 neither local qualification nor issued current scope')
        return original_pad(e, *args, **kwargs)

    native.NativePearlEngine._pad_spec_rhythm_target_tree_graph = guarded_pad
    engine._canonical235_pad = guarded_pad
    engine._canonical235_actual_calls = {}
    original_forward = native.NativePearlEngine.target_tree_forward

    def observed_forward(e, *args, **kwargs):
        result = original_forward(e, *args, **kwargs)
        if e._canonical235_phase == 'qualified':
            generation = getattr(e, '_dynamic_tree_generation', None)
            if type(generation) is not int:
                raise RuntimeError('ROOT235 real production generation required')
            state = e._branch_tail_state
            _, _, meta, enabled = state.last
            budgets = result['num_draft_tokens']
            record = dict(N=len(budgets), C=sum(budgets), M=result['query_count'],
                          logical_queries=result['logical_query_count'], branches=budgets.count(5),
                          attention_segments=len(meta.actual_seq_lengths_q),
                          mask_capacity=int(meta.tree_attention_mask.shape[2]),
                          copied=enabled, actual_copy_slots=state.last_record.get('real_copy_slots', 0))
            if not (1 <= record['N'] <= 16 and record['C'] <= 64 and record['M'] == 80
                    and record['logical_queries'] == record['N']+record['C']
                    and record['copied'] is bool(record['branches'])
                    and record['mask_capacity'] == (10 if record['copied'] else 9)):
                raise RuntimeError('ROOT235 actual canonical production call violates effective contract')
            e._canonical235_actual_calls.setdefault(str(generation), []).append(record)
        return result

    native.NativePearlEngine.target_tree_forward = observed_forward
    engine._canonical235_forward = observed_forward
    folder = Path(output)/'canonical-independent-node-math'
    stage('ROOT235/create-private-math-output', lambda:folder.mkdir(exist_ok=False)
          if engine.rank == engine.topology.draft_leader_rank else None)
    fixture = json.loads((P/'FIXTURE234.json').read_text())
    from bank231 import build
    bank = build(engine, native, stage, fixture['prefixes'])
    from neural_oracle231 import run
    math = run(engine, native, stage, bank, output=folder)
    stage('ROOT235/release-independent-bank', lambda:engine._release_cache() if not engine.is_draft else None)
    proof = dict(activation=activation, bank=bank['evidence'], math=math,
                 original_sampler_unchanged=True, original_component=original,
                 cases=159, production_qualified=False, whole_workload_AR_qualified=False)
    from runtime235 import save
    stage('ROOT235/retain-current-independent-math', lambda:save(folder, f'rank{engine.rank}-proof.json', proof))
    engine._canonical235_actual_proof = proof
    # Keep original component object identities and audits intact.
    return original


def cache_snapshot():
    import extension218
    return extension218.cache_snapshot()


def postfull_AR(engine, native, stage, windows, output):
    from reference235 import run
    return run(engine, native, stage, windows, output)


def snapshot(engine):
    return dict(actual_calls_by_generation=engine._canonical235_actual_calls,
                effective_requests=16, effective_candidates=64, physical_queries=80,
                configured_requests=32, configured_candidates=96,
                legacy218_upgrade_saved_physical_tokens_counter_is_counterfactual_only=True)
