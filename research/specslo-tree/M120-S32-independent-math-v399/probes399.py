"""Fresh S32/canonical-S24 strict math and actual same-engine graph costs."""
import json
from pathlib import Path
import statistics
import time


def save(folder, name, value):
    with (Path(folder)/name).open('x') as stream:
        json.dump(value, stream, separators=(',', ':'), allow_nan=False)


def graph_costs(engine, stage, bank, extent):
    from domain399 import cases
    from oracle399 import candidate, prepare
    from frames255 import pack
    from dataclasses import replace
    import torch
    case = next(c for c in cases() if c['name'] == 'N32-saturated')
    plans, rows = prepare(case, bank['prefixes'])
    stage('cost/actual32-full-math-refresh', lambda:candidate(engine, case, bank, plans, rows)
          if not engine.is_draft else None)
    frames = {}

    def prepare_frames():
        if engine.is_draft:
            return None
        x, pos, metadata, _, _ = engine._branch_tail_state.static_current['physical']
        frames['S32'] = (x, pos, metadata)
        owners = list(range(24))
        positions = [len(bank['prefixes'][i]) for i in owners]
        tokens = [[bank['roots'][i], *[103+17*i+29*j for j in range(4)]] for i in owners]
        xx, pp, mm = pack(engine, owners, positions, tokens, rows=120, reference=False)
        frames['S24'] = (xx, pp, replace(mm, sequence_lens=(extent,)*24))
        return dict(S24_M=120, S32_M=120, S24_queries=[5]*24, S32_queries=[4]*24+[3]*8,
                    visible_extent=extent, input_frames_prepared_outside_timer=True,
                    model_weights_and_MC2_profile_unchanged=True,
                    scope='prepared physical graph calls, not full verification or serving latency')

    description = stage('cost/independent-physical-frames', prepare_frames)

    def invoke(kind, timed):
        if engine.is_draft:
            return None
        x, pos, meta = frames[kind]
        torch.npu.synchronize()
        started = time.perf_counter()
        output = engine.graph_runner.run_target_greedy([x], [pos], [meta], engine.draft_vocab_size)
        torch.npu.synchronize()
        elapsed = (time.perf_counter()-started)*1000
        actual = engine.graph_runner.last_target_execution
        if not actual.used_aclgraph or actual.mode not in (('replay',) if timed else ('capture_replay', 'replay')):
            raise RuntimeError('ROOT399 actual warm/replayed graph cost path required')
        if len(output) != 1 or output[0].shape[0] != 120:
            raise RuntimeError('ROOT399 complete M120 graph output required')
        return elapsed

    for repeat in range(4):
        for kind in ('S24', 'S32'):
            stage(f'cost/warm{repeat}/{kind}', lambda kind=kind:invoke(kind, False))
    values = {'S24': [], 'S32': []}
    for repeat in range(20):
        for kind in (('S24', 'S32') if repeat % 2 == 0 else ('S32', 'S24')):
            values[kind].append(stage(f'cost/pair{repeat}/{kind}', lambda kind=kind:invoke(kind, True)))
    if engine.is_draft:
        return dict(target_cost_measured=False)
    return dict(description=description, samples_ms=values,
                median_ms={k:statistics.median(v) for k,v in values.items()},
                p95_ms={k:sorted(v)[18] for k,v in values.items()},
                full463=False, Raw=None, Goodput=None)


def run(engine, native, stage, output, fixture):
    from bank399 import build
    from domain399 import cases, prefix_bank
    from oracle399 import run as qualify
    from static_state399 import install
    from startup_receipts_v75 import NATIVE_COUNTERS, FAILURE_COUNTERS
    if len(cases()) != 40:
        raise RuntimeError('ROOT399 explicit40-case finite geometry')
    target = not engine.is_draft
    activation = stage('ROOT399/activate-finite-S32', lambda:install(engine, native))
    reports = []
    for extent in (512, 1024, 1536, 2048, 2560):
        lane = lambda name, callback:stage(f'ROOT399/KV{extent}/'+name, callback)
        bank = build(engine, native, lane, prefix_bank(fixture['prefixes'], extent))
        folder = Path(output)/f'KV{extent}'
        lane('private-output', lambda:folder.mkdir(exist_ok=False) if engine.rank == 0 else None)
        before = len(engine._branch_tail_state.static_graph_calls)
        math = qualify(engine, native, lane, bank, folder)

        def strict():
            if not target:
                return None
            if not (len(math['cases']) == 40 and math['all_independent_prefixes_unchanged'] is True
                    and all(all(row[k] is True for k in (
                        'exact_greedy', 'hidden_bitwise_equal', 'KV_bitwise_equal',
                        'graph_eager_KV_bitwise_equal', 'allphysical_graph_eager_hidden_KV'))
                        and row['actual_physical_S'] == 32 and row['independent_reference_S'] == 24
                        for row in math['cases'])):
                raise RuntimeError('ROOT399 all new strict node/hidden/allKV cases before cost')
            return True

        lane('all40-bitwise-before-cost', strict)
        calls = list(engine._branch_tail_state.static_graph_calls[before:])
        costs = graph_costs(engine, lane, bank, extent)

        def protected():
            if not target:
                return None
            from dual_bank_v146 import prefix_evidence
            for owner, expected in enumerate(bank['evidence']['all_prefix_evidence']):
                if prefix_evidence(engine, {owner:bank['prefixes'][owner%32]}) != expected:
                    raise RuntimeError('ROOT399 all96 original prefixes protected after math/cost')
            return True

        prefix_proof = lane('all96-prefixes-after-cost', protected)
        value = dict(extent=extent, math=math, actual_static_graph_calls=calls,
                     independent_bank=bank['evidence'], target_graph_cost=costs,
                     all96_prefixes_protected=prefix_proof, serving_qualified=False)
        lane('retain-complete-bank', lambda:save(folder, f'rank{engine.rank}-bank399.json', value))
        reports.append(value)
        lane('release-bank', lambda:engine._release_cache() if target else None)
    metrics = {key:engine.graph_metrics()[key] for key in NATIVE_COUNTERS}
    if any(metrics[key] for key in FAILURE_COUNTERS):
        raise RuntimeError('ROOT399 no Native failure or fallback counters')
    return dict(rank=engine.rank, activation=activation, banks=reports,
                actual_model_cases=200 if target else 0, target_S=32, reference_S=24, physical_M=120,
                Native_final=metrics, original_full_components_executed=False,
                whole_workload_AR_qualified=False, full463=False, performance=False,
                production_permission=False, Raw=None, Goodput=None, adopted=False)
