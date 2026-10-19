"""Fresh finite target math, then same-instance paired costs; no live service."""
from pathlib import Path
import json
import time
from geometry336 import LAYOUTS, cases, cost_arms, make_plans, teacher_rows


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT336: '+message)


def native_call(engine, bank, *, layout, owners, rows=None):
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    width, groups = LAYOUTS[layout]
    engine._q4_layout336 = layout
    n = len(owners)
    plans = make_plans(bank['prefixes'], owners, width)
    roots = [bank['roots'][i] for i in owners]
    draft = teacher_rows(owners, width) if rows is None else [list(row) for row in rows]
    ids = list(owners)
    for j in range(groups-n):
        plans.append(cached_cpu_tree_speculation_plan(1, width-1, 0, 4096, candidate_budget=width-1))
        roots.append(0)
        draft.append([0]*(width-1))
        ids.append(463+j)
    return engine.target_tree_forward(plans, roots, draft, sequence_ids=ids,
        return_logits=False, real_tree_count=n)


def candidate(engine, case, bank):
    import torch
    from dual_bank_v146 import query_KV
    rows = teacher_rows(case['owners'], case['width'])
    out = native_call(engine, bank, layout=case['layout'], owners=case['owners'], rows=rows)
    q = case['query_count']
    need(out['used_aclgraph'] is True and out['used_tree_attention'] is True
         and out['query_count'] == case['M'] and out['logical_query_count'] == q
         and out['num_draft_tokens'] == case['budgets'], 'actual complete Native FULL target call')
    tokens = out['target_query_token_ids'].clone()
    graph_KV = query_KV(engine, out['cache_slot_mapping'].long())
    x, positions, metadata = engine._q4_last_inputs336
    need(x.numel() == case['M'], 'eager companion uses actual captured inputs')
    hidden = engine.model(x, positions, metadata).clone()
    eager_tokens = engine.model.compute_greedy_tokens(hidden, engine.draft_vocab_size)[:q]
    eager_KV = query_KV(engine, metadata.slot_mapping[:q].long())
    need(bool(torch.isfinite(hidden).all()) and torch.equal(tokens, eager_tokens), 'all real graph/eager heads')
    need(engine.graph_runner._target_outputs_match(tuple(graph_KV), tuple(eager_KV)), 'unchanged all128 Native KV matcher')
    exact = all(torch.equal(a, b) for a, b in zip(graph_KV, eager_KV, strict=True))
    need(exact, 'all128 graph/eager query KV bitwise')
    return dict(tokens=tokens, hidden=hidden[:q].clone(), KV=graph_KV,
                graph_eager_KV_bitwise_equal=exact)


def reference(engine, case, bank, stage, *, future):
    import torch
    from dual_bank_v146 import query_KV
    from metadata336 import reference_frame
    width = case['width']
    target = not engine.is_draft
    rows = teacher_rows(case['owners'], width)
    tokens, hidden_rows, KV_rows = {}, {}, {}
    frames = []
    for depth in range(width):
        def execute():
            if not target:
                return None
            positions = [len(bank['prefixes'][owner])+depth for owner in case['owners']]
            inputs = [bank['roots'][owner] if depth == 0 else row[depth-1]
                      for owner, row in zip(case['owners'], rows, strict=True)]
            x, pos, meta = reference_frame(engine, layout=case['layout'],
                owners=[32+owner for owner in case['owners']], positions=positions,
                tokens=inputs, future=future)
            hidden = engine.model(x, pos, meta).clone()
            need(bool(torch.isfinite(hidden).all()), 'finite independent own-history hidden')
            heads = engine.model.compute_greedy_tokens(hidden, engine.draft_vocab_size)
            head_rows = [i*width for i in range(case['N'])]
            index = torch.tensor(head_rows, dtype=torch.long, device=engine.device)
            kv = query_KV(engine, meta.slot_mapping.index_select(0, index).long())
            selected = heads.index_select(0, index).cpu().tolist()
            for i, value in enumerate(selected):
                destination = i*width+depth
                need(destination not in tokens, 'each independent node evaluated once')
                tokens[destination] = value
                hidden_rows[destination] = hidden[head_rows[i]].clone()
                KV_rows[destination] = [part[i].clone() for part in kv]
            return dict(depth=depth, real_nodes=case['N'], physical_queries=case['M'],
                future=future, candidate_KV_copied=False, reference_eager_only=True)
        frames.append(stage(f'reference-canary{future}/depth{depth}', execute))
    if not target:
        return None
    q = case['query_count']
    need(sorted(tokens) == sorted(hidden_rows) == sorted(KV_rows) == list(range(q)), 'all independent real nodes present')
    return dict(tokens=torch.tensor([tokens[i] for i in range(q)], dtype=torch.long, device=engine.device),
        hidden=torch.stack([hidden_rows[i] for i in range(q)]),
        KV=[torch.stack([KV_rows[i][layer] for i in range(q)]) for layer in range(128)], frames=frames)


def paired_cost(engine, bank, stage):
    import torch
    arms = cost_arms()
    target = not engine.is_draft
    expected = {}
    for arm in arms:
        def warm():
            if not target:
                return None
            out = native_call(engine, bank, layout=arm['layout'], owners=list(range(arm['N'])))
            torch.npu.synchronize()
            return out['target_query_token_ids'].cpu().tolist()
        expected[arm['name']] = stage('paired-cost/'+arm['name']+'/warm', warm)
    before = engine.graph_metrics()
    pairs = []
    for iteration in range(20):
        ordered = arms[iteration % len(arms):]+arms[:iteration % len(arms)]
        samples = {}
        for arm in ordered:
            def execute():
                if not target:
                    return None
                torch.npu.synchronize()
                started = time.perf_counter()
                out = native_call(engine, bank, layout=arm['layout'], owners=list(range(arm['N'])))
                torch.npu.synchronize()
                elapsed = (time.perf_counter()-started)*1000
                need(out['used_aclgraph'] is True, 'paired measurement uses original ACLGraph')
                need(out['target_query_token_ids'].cpu().tolist() == expected[arm['name']], 'unchanged repeated greedy head')
                return elapsed
            samples[arm['name']] = stage(f'paired-cost/rotation{iteration}/'+arm['name'], execute)
        pairs.append(dict(order=[arm['name'] for arm in ordered], milliseconds=samples))
    after = engine.graph_metrics()
    keys = ('aclgraph_captures', 'aclgraph_capture_attempts', 'aclgraph_runtime_validation_replays',
            'aclgraph_failed_captures', 'aclgraph_capacity_fallbacks', 'aclgraph_shape_fallbacks')
    need(all(before[k] == after[k] for k in keys), 'no new graph captures/validation/fallbacks in paired costs')
    return dict(arms=arms, pairs=pairs, rounds=20, graph_counters_before=before,
        graph_counters_after=after, no_new_graph_work=True,
        scope='synchronous explicit fixed-frame padding plus original Native target setup/metadata/model/output/sync; comparison D2H and stage consensus excluded',
        real_draft_generation=False, complete_service=False, Goodput=None)


def run(engine, native, stage, output, fixture):
    import torch
    from metadata336 import install, ensure_zero
    from bank326 import build
    from dual_bank_v146 import prefix_evidence
    target = not engine.is_draft
    activation = stage('ROOT336/install-private-fixed-frame-metadata', lambda: install(engine))
    bank = build(engine, native, stage, fixture['prefixes'])
    stage('ROOT336/independent-zero-page', lambda: ensure_zero(engine) if target else None)
    model_metrics_before = engine.graph_metrics()
    folder = Path(output)/'independent-node-math'
    stage('ROOT336/create-math-output', lambda: folder.mkdir(exist_ok=False) if engine.rank == 0 else None)
    records = []
    for case in cases():
        lane = lambda label, fn: stage('ROOT336/'+case['name']+'/'+label, fn)
        actual = lane('actual-Native-graph-and-eager', lambda: candidate(engine, case, bank) if target else None)
        comparisons = []
        for future in (0, 17):
            own = reference(engine, case, bank, lane, future=future)
            def compare():
                if not target:
                    return None
                matcher = engine.graph_runner._target_outputs_match
                need(torch.equal(actual['tokens'], own['tokens']), 'all greedy heads equal independent nodes')
                need(matcher((actual['hidden'],), (own['hidden'],)), 'original hidden matcher unchanged')
                need(matcher(tuple(actual['KV']), tuple(own['KV'])), 'all128 independent KV matchers unchanged')
                hidden_exact = torch.equal(actual['hidden'], own['hidden'])
                kv_exact = all(torch.equal(a, b) for a, b in zip(actual['KV'], own['KV'], strict=True))
                need(hidden_exact and kv_exact, 'strict independent hidden and all128KV bitwise equality')
                return dict(future=future, exact_greedy=True, hidden_bitwise_equal=hidden_exact,
                    all128_KV_bitwise_equal=kv_exact, graph_eager_KV_bitwise_equal=actual['graph_eager_KV_bitwise_equal'],
                    candidate_tokens=actual['tokens'].cpu().tolist(), reference_tokens=own['tokens'].cpu().tolist(),
                    frames=own['frames'], candidate_KV_copied=False)
            comparisons.append(lane('compare-canary'+str(future), compare))
        record = dict(case=case, comparisons=comparisons) if target else None
        records.append(record)
        def retain():
            value = dict(rank=engine.rank, case=case, observation=record,
                         neural_math_on_rank=target, performance_qualified=False)
            with (folder/f'rank{engine.rank}-case{len(records):03d}.json').open('x') as saved:
                saved.write(json.dumps(value, allow_nan=False)+'\n')
        lane('retain-completed-case', retain)
    cold_calls = list(engine._q4_native_calls336) if target else []
    costs = paired_cost(engine, bank, stage)
    def protected():
        if not target:
            return None
        for owner, expected in enumerate(bank['evidence']['all_prefix_evidence']):
            need(prefix_evidence(engine, {owner: bank['prefixes'][owner % 32]}) == expected,
                 'all96 independently committed prefixes protected')
        return True
    prefixes = stage('ROOT336/all96-prefixes-after-math-and-cost', protected)
    model_metrics_after = engine.graph_metrics()
    stage('ROOT336/release-bank', lambda: engine._release_cache() if target else None)
    return dict(rank=engine.rank, activation=activation, bank=bank['evidence'], cases=records,
        case_count=len(records), cold_native_calls=cold_calls, paired_target_cost=costs,
        original_engine_metrics_before=model_metrics_before, original_engine_metrics_after=model_metrics_after,
        all_prefixes_exact_after_cost=prefixes, real_draft_generation=False,
        whole_workload_AR_qualified=False, performance=False, adopted=False)
