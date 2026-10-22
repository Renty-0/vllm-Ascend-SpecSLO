"""Independent future189 terminal evidence audit. Never launch model/device work.

ROOT must authorize a terminal receipt by its exact SHA before any actual read.
Only standard-library helpers from the completed173 auditor are AST-reused.
"""
import argparse, ast, collections, copy, datetime, gzip, hashlib, importlib.abc
import json, math, re, statistics, sys, tarfile, types
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-slo-admission1-causal80-active32-full-v189'
ACTION = 'slo-admission1-causal80-active32-full463'
FREEZE_SHA = 'ac0233ffafcc99b8303c8b88c4911cf4c2d0cc7d4129ff1b6ac6ba879c9777eb'
ROOT_PREPARED = D / 'ROOT_V189_PREPARED_SOURCE_HOST_AUDIT.json'
ROOT_PREPARED_SHA = 'f9b9032cfad346a7e45c1ee0db6f84386060c1da1cc86182f1154d18fe7b91c1'
HELPERS = D / 'audit_v173_terminal_full463_HOST.py'
HELPER_SHA = 'ed8aef51646ccf60c4812782d080d119fa406d7e53138ce24d6b470bce74c6f5'
CLOCKS = D / 'analyze_completed_tree_full463_cost.py'
CLOCK_SHA = 'd3266647799fc0cf00c49fe98e195ecb85dc446e51ab66c643a5b1d5c3ca0b74'
NATIVE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
BACKENDS = ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'transformers', 'safetensors', 'acl')

if hashlib.sha256(HELPERS.read_bytes()).hexdigest() != HELPER_SHA:
    raise RuntimeError('Exact completed173 stdlib helper source changed')
_names = {'need', 'sha', 'desc', 'read', 'archive', 'Firewall', 'extracted', 'Value', 'pct'}
_nodes = [copy.deepcopy(n) for n in ast.parse(HELPERS.read_text()).body
          if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in _names]
if {n.name for n in _nodes} != _names:
    raise RuntimeError('Complete selected173 stdlib helper AST')
exec(compile(ast.fix_missing_locations(ast.Module(body=_nodes, type_ignores=[])),
             str(HELPERS) + '::isolated189-stdlib-helpers', 'exec'), globals())


def provenance(receipt):
    freeze_path = P / 'FROZEN_SOURCE189.json'
    need(sha(freeze_path) == FREEZE_SHA and sha(ROOT_PREPARED) == ROOT_PREPARED_SHA,
         'exact frozen189 and independent ROOT preparation')
    f = read(freeze_path); pins = {}
    for key in ('files', 'external_files', 'retained_HOST_evidence', 'execution_guard_inputs', 'peer_reviews'):
        for path, value in f[key].items():
            need(path not in pins or pins[path] == value, 'consistent frozen input ' + path)
            pins[path] = value
    pins.update({str(P / 'CPU_RESULT189.json'): f['CPU_sha256'], str(freeze_path): FREEZE_SHA,
                 f['root_review']['path']: f['root_review']['sha256']})
    need(len(pins) == 253 and all(sha(path) == value for path, value in pins.items()),
         'actual253 exact selected source/evidence bytes')
    cpu = read(P / 'CPU_RESULT189.json'); prepared = read(P / 'PREPARED_HOST189.json')
    need(cpu['schema'] == 'V189.actual-admission-policy-source-HOST.v1' and cpu['status'] == 'passed'
         and cpu['tests_run'] == 3 and cpu['errors'] == cpu['failures'] == 0
         and cpu['Torch_imported'] is cpu['NPU_used'] is cpu['model_constructed'] is
         cpu['original_parent_suite_rerun'] is cpu['device_qualification_inherited'] is False,
         'actual own3 CPU producer fields, no invented backend field')
    need(cpu['local_before'] == cpu['local_after'] == f['files'] == prepared['files'] and
         cpu['external_before'] == cpu['external_after'] == f['external_files'] == prepared['external_files']
         and sha(P / 'CPU_RESULT189.json') == f['CPU_sha256'] == prepared['CPU_sha256'],
         'exact CPU before/after/prepared/freeze binding')
    expected_scope = dict(SLO_AWARE_ADMISSION=True, TARGET_SLO_FIRST=False, prefill_coalesce=[1, 0.0],
                          active_windows_requested=[32], configured_capacity=64, queued=463, storage=527)
    need(f['ROOT_source_reviewed'] is True and all(f[k] == v for k, v in expected_scope.items())
         and all(receipt[k] == v for k, v in expected_scope.items()), 'source and actual single-policy arm scope')
    root = read(f['root_review']['path'])
    reviewed = {k: v for k, v in pins.items() if k not in (str(freeze_path), f['root_review']['path'])}
    need(root['status'] == 'passed_source_review' and root['reviewed_inputs'] == reviewed,
         'ROOT exact251 reviewed source/CPU/guard/peer map')
    readiness = receipt['readiness']
    need(readiness['freeze_sha256'] == FREEZE_SHA and readiness['CPU_sha256'] == f['CPU_sha256']
         and readiness['root_review'] == f['root_review'] and readiness['peers'] == f['peer_reviews']
         and readiness['device_qualification_inherited'] is False, 'actual original guard readiness')
    ar = readiness['archive']
    need(Path(ar['path']) == P / 'PREPARED_SOURCE_ARCHIVE189.tar.gz' and ar['members'] == 253
         and sha(ar['path']) == ar['sha256'], 'actual own253 archive identity')
    own_archive = archive(ar['path'], pins)
    # These immutable parent binaries are verified in place, never copied.
    import base189, check189
    dependencies = dict(original=base189.parent_archive(), production_parent173=check189.parent_frozen(archive=True))
    need(dependencies == f['dependency_archives_verified_in_place'] == readiness['dependency_archives'],
         'P1486390/P16042/P16677/P173158 exact original archives in place')
    safe = D / 'v177-root-safe-storage-overlay/safe_storage177.py'
    need(str(safe) in f['external_files'] and sha(safe) == f['external_files'][str(safe)],
         'exact frozen177 scoped storage implementation, not a separate189 overlay run')
    for key in ('pre_verifier', 'post_verifier'):
        row = receipt[key]; value = read(row['path'])
        need(sha(row['path']) == row['sha256'] and row['exit_code'] == 0 and value == row['actual_result']
             and value['status'] == 'pass' and value['errors'] == [] and
             (value['checked_frozen_files'], value['checked_external_files'], value['checked_packages']) ==
             (10496, 15, 212), 'retained original10496/15/212 ' + key)
    for key in ('devices_before', 'devices_after'):
        value = receipt[key]
        need(value['all_idle'] is value['four_910B2'] is True and set(value['processes']) == {'2', '5', '6', '7'},
             'saved all4 idle ' + key)
        need(all(v['exit_code'] == 0 and v['idle'] is True and not v.get('timed_out', False)
                 and re.search(r'No process in device\.|No running processes found', v['output'], re.I)
                 for v in value['processes'].values()), 'saved original idle outputs ' + key)
    need(receipt['physical_cards'] == [2, 5, 6, 7] and receipt['sources_unchanged'] is True
         and receipt['owned_remaining'] == [] and receipt['execution_lock']['path'] == str(B / '.ROOT_NPU_EXECUTION.lock')
         and receipt['execution_lock']['nonblocking_exclusive'] is True, 'self ROOT lock/source/owned cleanup')
    limits = dict(child_execution_seconds=7200, actual_tag_bytes=256 * 1024 ** 2,
                  aggregate_private_bytes=512 * 1024 ** 2)
    need(receipt['limits'] == limits and math.isfinite(receipt['elapsed_seconds'])
         and 0 < receipt['elapsed_seconds'] < 7200, 'original actual bounded child duration')
    storage = receipt['private_storage_final']
    need(0 <= storage['actual_tag_bytes'] <= limits['actual_tag_bytes'] and
         0 <= storage['aggregate_private_bytes'] <= limits['aggregate_private_bytes'], 'saved final private storage')
    need(sha(CLOCKS) == CLOCK_SHA, 'exact independent original clocks helper')
    return dict(freeze=desc(freeze_path), CPU=desc(P / 'CPU_RESULT189.json'), ROOT_prepared=desc(ROOT_PREPARED),
                selected_current_SHA_checked=253, archive=own_archive, parent_archives=dependencies,
                storage177_source=desc(safe), saved_prepost='10496/15/212 PASS; not rerun',
                saved_all4_idle=True, owned_remaining=[], complete_environment_import_closure_claimed=False)


def component_integer_verdicts(directory):
    """Run unchanged Native integer verifier against saved heads, not model math."""
    ht = types.SimpleNamespace(Tensor=Value, tensor=lambda v, **kw: Value(v), long='long')
    ns = dict(torch=ht, TreeVerificationOutput=lambda a, b: types.SimpleNamespace(token_ids=a, accepted_node_indices=b))
    verify = extracted(NATIVE / 'native_engine.py', '_verify_tree_outputs_host', ns)
    counts = collections.Counter(); cross = []
    for folder_name, passes_per_rank in (('variable-B4-own-full-process', 6), ('causal80-own-current-process', 3)):
        folder = directory / folder_name
        for rank in range(4):
            report = read(folder / f'rank{rank}-component.json.gz')
            need(report['rank'] == rank and len(report['traces']) == passes_per_rank, 'actual role/pass inventory')
            for trace in report['traces']:
                value = trace['Native']; before, after, delta = (value[k] for k in ('before', 'after', 'delta'))
                need(len(delta) == 21 and set(before) == set(after) == set(delta) and
                     all(type(before[k]) is type(after[k]) is type(delta[k]) is int
                         and 0 <= before[k] <= after[k] and after[k] - before[k] == delta[k] for k in delta),
                     'independent actual21 monotone subtraction')
                if trace['zero_replay']:
                    need(not any(delta.values()), 'entire protocol no-new-work zero')
                    counts[folder_name + '_zero_subtractions'] += 21
            if rank == 0:
                continue
            for phase in range(3):
                cases = range(2) if folder_name.startswith('variable') else (None,)
                for case in cases:
                    for cycle in range(4):
                        suffix = f'case{case}-' if case is not None else ''
                        row = read(folder / f'rank{rank}-pass{phase}-{suffix}cycle{cycle}.json.gz')
                        budgets = [len(v) for v in row['rows']]
                        plans = [types.SimpleNamespace(candidate_budget=v, parent_indices=Value([-1, 0, 1, 2][:v])) for v in budgets]
                        arms = ('old', 'new') if case is None else ('reference',)
                        for arm in arms:
                            head = row['crossM'][arm + '_real80'] if case is None else row['reference']['real_query_tokens']
                            out = verify(row['rows'], plans, Value(head), 4, 'explicit-HOST-values')
                            tokens = [[v for v in r if v >= 0] for r in out.token_ids.values]
                            expected_tokens = row['crossM'][arm + '_verdict_tokens'] if case is None else row['commit']['committed_tokens']
                            expected_paths = row['crossM'][arm + '_accepted'] if case is None else row['commit']['accepted_paths']
                            need(tokens == [[v for v in r if v >= 0] for r in expected_tokens]
                                 and out.accepted_node_indices.values == expected_paths,
                                 'unchanged Native verifier on actual saved ' + arm + ' heads')
                            counts[folder_name + '_integer_verifier_calls'] += 1
                            counts[folder_name + '_owner_verdicts'] += len(budgets)
                        if case is None:
                            cm = row['crossM']; different = [i for i, (a, b) in enumerate(zip(cm['old_real80'], cm['new_real80'], strict=True)) if a != b]
                            need(len(cm['old_real80']) == len(cm['new_real80']) == 80 and different == cm['different_head_positions']
                                 and cm['head_equal'] == (not different), 'retained cross-M complete80 positions')
                            cross.append(dict(rank=rank, phase=phase, cycle=cycle, different_head_positions=different,
                                              verdict_equal=cm['verdict_equal'], accepted_equal=cm['accepted_equal']))
    need(counts['variable-B4-own-full-process_zero_subtractions'] == 336 and
         counts['causal80-own-current-process_zero_subtractions'] == 168 and
         counts['variable-B4-own-full-process_integer_verifier_calls'] == 72 and
         counts['causal80-own-current-process_integer_verifier_calls'] == 72 and len(cross) == 36,
         'all504 component zeros plus144 integer verifier calls')
    need([v for v in cross if v['rank'] == 1] == [dict(v, rank=1) for v in cross if v['rank'] == 2] ==
         [dict(v, rank=1) for v in cross if v['rank'] == 3], 'three target ranks cross-M outcomes agree')
    return dict(counts=counts, current80_crossM_records=cross,
                all80_heads_equal=all(not v['different_head_positions'] for v in cross),
                all_verdicts_equal=all(v['verdict_equal'] and v['accepted_equal'] for v in cross),
                hidden_head_KV_tensors_recomputed_on_HOST=False)


def admission_and_full_statistics(traces, reports):
    from admission189 import audit_trace
    for rank, trace in enumerate(traces):
        audit_trace(trace, 463)
        need(trace['sealed_before'] is trace['sealed_after'] is trace['cache_released'] is True,
             'all4 sealed and released full')
        need(trace['configured_capacity'] == 64 and trace['admission_window160'] == 32
             and 0 < trace['active_peak'] <= 32 and trace['actual_capacity64_scope_completed'] is False,
             'truthful actual active32 vs configured64')
        need(reports[rank]['admission_policy']['coalesce_min_requests'] == 1 and
             reports[rank]['admission_policy']['coalesce_max_wait_ms'] == 0.0, 'actual current cfg1/0')
    t = traces[1]; ns = dict(math=math, statistics=statistics, CLASSES=('tight', 'normal', 'loose'))
    extracted(CLOCKS, 'need', ns); metrics = extracted(CLOCKS, 'clocks', ns)(t)
    first = {}
    for e in t['stream_events']:
        owner = e['request_index']
        if owner not in first:
            need('first_token_wall_seconds160' in e, 'actual first worker-submission timestamp')
            first[owner] = e['first_token_wall_seconds160']
        else:
            need('first_token_wall_seconds160' not in e, 'one first timestamp per original request')
    need(set(first) == set(range(463)) and all(row['arrival_ts'] == t['absolute_arrivals'][i]
         for i, row in enumerate(t['generate_results'])), 'all463 original arrival clocks and first callbacks')
    ttft = [(first[i] - row['arrival_ts']) * 1000 for i, row in enumerate(t['generate_results'])]
    need(all(math.isfinite(v) and v >= 0 for v in ttft), 'finite nonnegative worker TTFT')
    metrics['TTFT_worker_submission'] = dict(samples=463, **pct(ttft), by_class={cl: pct([
        ttft[i] for i, row in enumerate(t['generate_results']) if row['slo_class'] == cl]) for cl in ('tight', 'normal', 'loose')})
    retained = next(v['output'] for v in reports[1]['windows']['per_rank_full_metrics'] if v['window'] == 32)
    need(all(retained[k] == v for k, v in metrics.items() if k != 'TTFT_worker_submission') and
         all(retained['TTFT160'][k] == metrics['TTFT_worker_submission'][k] for k in ('median_ms', 'P90_ms')),
         'independent original clocks/TTFT match actual runtime metrics')
    calls = t['normal_target_calls']; owners = sum(c['N'] for c in calls)
    nodes = sum(sum(c['budgets']) for c in calls); real = sum(c['real_query_count'] for c in calls)
    physical = sum(c['physical_M'] for c in calls); dc = t['decode_counters']
    need(nodes == dc['spec_rhythm_verified_tokens'] and physical == dc['spec_rhythm_target_query_tokens']
         and physical - real == dc['spec_rhythm_target_graph_padding_queries'], 'real target geometry counters')
    selected80 = [c for c in calls if (c['N'], c['real_query_count'], c['family']) == (16, 80, 'causal')]
    need(all((c['physical_M'], c['physical_S'], c['physical_query_partition']) ==
             (80, 16, [v + 1 for v in c['budgets']]) for c in selected80), 'actual current80 mapper usage')
    prefill_ids = [owner for c in t['standalone_prefill_calls'] for owner in c['sequence_ids']]
    need(sorted(prefill_ids) == list(range(463)) and sum(c['uncached_queries'] for c in t['standalone_prefill_calls']) ==
         sum(len(v) for v in t['tokenized_inputs']) and t['logical_events'] == [], 'all463 genuine standalone whole prompts, no drop or mixed claim')
    queues = collections.defaultdict(collections.deque)
    for c in calls:
        for owner, budget in zip(c['sequence_ids'][:c['N']], c['budgets'], strict=True): queues[owner].append(budget)
    initial = set(); delivered = visible_bonus = 0; align_error = None
    for e in t['stream_events']:
        owner = e['request_index']
        if owner not in initial:
            need(len(e['token_ids']) == 1, 'one original prefill/root token')
            initial.add(owner); continue
        delivered += len(e['token_ids'])
        if queues[owner]: visible_bonus += int(len(e['token_ids']) == min(queues[owner].popleft(), 4) + 1)
        else: align_error = 'A post-initial callback has no retained target owner occurrence.'
    if any(queues.values()): align_error = 'Some retained target owner occurrences have no callback.'
    bonus = dict(owner_rounds=owners, accepted_nodes=dc['spec_rhythm_tree_accepted_nodes'],
                 Native_full_primary_accepts=dc['spec_rhythm_tree_full_accepts'], delivered_decode_tokens=delivered,
                 verified_event_alignment_exact=align_error is None, alignment_limitation=align_error)
    if align_error is None:
        accepted = bonus['accepted_nodes']; full = bonus['Native_full_primary_accepts']; discarded = accepted + owners - delivered
        need(discarded >= full - visible_bonus >= 0, 'primary bonus/truncation accounting')
        bonus.update(accepted_nodes_per_owner=accepted / owners, delivered_decode_per_owner=delivered / owners,
                     visible_primary_bonus=visible_bonus, discarded_primary_bonus=full - visible_bonus,
                     total_output_limit_discarded=discarded, subtraction_is_not_a_rerun_without_bonus=True)
    pages = {}
    for trace in traces:
        a = trace['allocator']; need(a['pool_blocks'] == 544 and a['allocation_storage_rows'] == [527] and
            0 <= a['referenced_page_highwater'] <= 544 and a['final_referenced_pages'] == 0 and
            a['acquired_pages'] == a['released_pages'], 'actual page highwater/release, not HBM bytes')
        pages[str(trace['rank'])] = a
    return dict(metrics=metrics, target_calls=len(calls), owner_rounds=owners, mean_N=owners / len(calls),
                budget_owner_rounds=dict(sorted(collections.Counter(v for c in calls for v in c['budgets']).items())),
                N_histogram=dict(sorted(collections.Counter(c['N'] for c in calls).items())),
                current80_calls=len(selected80), physical_queries=physical, real_queries=real, padding_queries=physical-real,
                standalone_prefill_calls=len(t['standalone_prefill_calls']), all463_prefill_owner_ids_exact=True,
                actual_SLO_admission1_target0_coalesce1_0=True, bonus_accounting=bonus, allocator_by_rank=pages,
                HOST_phase_seconds_by_role={str(v['rank']): v['phase_seconds'] for v in traces},
                clock_scope='Native charged decode and worker-submitted stream events, not external client clocks')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--terminal-authorized', action='store_true', required=True)
    parser.add_argument('--tag', required=True); parser.add_argument('--expected-receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.tag) is not None, 'safe exact tag')
    need(args.output.parent.resolve() == D and not args.output.exists(), 'fresh privateD output')
    receipt_path = P / ('receipt-' + args.tag + '.json')
    need(sha(receipt_path) == args.expected_receipt_sha256, 'ROOT explicit terminal189 receipt SHA')
    receipt = read(receipt_path)
    need(receipt['status'] in ('complete', 'failed') and receipt['action'] == ACTION, 'terminal original189 action only')
    actual = P / 'actual' / args.tag
    need(actual.is_dir() and actual.resolve() == actual and not actual.is_symlink(), 'private terminal189 actual')
    sys.meta_path.insert(0, Firewall()); sys.path.insert(0, str(P))
    errors = []; result = dict(schema='V189.independent-terminal-HOST-audit.v1', candidate=str(P), tag=args.tag,
        job_status=receipt['status'], receipt=desc(receipt_path), script=desc(__file__), errors=errors,
        Torch_imported=False, NPU_used=False, adopted=False, formal_Goodput=False)
    try:
        result['provenance'] = provenance(receipt)
        need(read(actual / 'worker-exit.json') == dict(closed=True, all_exited=True), 'all workers closed')
    except Exception as e: errors.append(dict(stage='source_or_terminal_provenance', error=repr(e)))
    paths = sorted(actual.rglob('*')); need(not any(p.is_symlink() for p in paths), 'no actual raw alias')
    manifest = {str(p): desc(p) for p in paths if p.is_file()}
    mp = args.output.with_name(args.output.stem + '_RAW_MANIFEST.json'); need(not mp.exists(), 'fresh raw manifest')
    mp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    result['raw_manifest'] = dict(**desc(mp), files=len(manifest))
    result['worker_errors'] = [read(p) for p in sorted(actual.glob('rank*-last-error.json'))]
    if 'provenance' in result:
        try:
            import audit189
            original = audit189.audit(actual, (32,)); complete = read(actual / 'completed.json')
            need(original['passed'] is True and original == complete['audit'] == receipt['actual_full_audit']
                 and complete['action'] == ACTION and complete['replies'] ==
                 [['calibrated', r, dict(windows=[32], full463=True)] for r in range(4)], 'original unmodified audit189 equals runtime/supervisor')
            result['original_unmodified_audit'] = dict(source=desc(P / 'audit189.py'), result=original)
            need(original['native_zero_subtractions'] == 252 and original['math_independent_qualified'] is True,
                 'all4 twozero plus sealed full21counter gates')
            need(receipt['status'] == 'complete' and receipt['errors'] == [] and receipt['child']['exit_code'] == 0
                 and receipt['actual_execution_complete'] is True and not result['worker_errors'], 'job success kept separate from retained inference')
            expected_status = 'passed_full' if original['old_padding_behavior_equal'] else 'passed_numerical_variant_full'
            need(receipt['actual_full_status'] == expected_status, 'truthful original compatibility vs numerical-variant status')
            result['additional_integer_component'] = component_integer_verdicts(actual)
        except Exception as e: errors.append(dict(stage='original_components_full_audit_and_job', error=repr(e)))
        try:
            traces = [read(actual / 'windows/window32/full463' / f'rank{r}-full463-actual-trace.json.gz') for r in range(4)]
            reports = [read(actual / f'rank{r}-local-diagnostic.json') for r in range(4)]
            result['retained_full463'] = admission_and_full_statistics(traces, reports)
        except Exception as e: errors.append(dict(stage='retained_full463_policy_arithmetic', error=repr(e)))
    need(not Firewall.attempts and not any(n.split('.')[0] in BACKENDS for n in sys.modules), 'no backend imports')
    result['status'] = 'passed_terminal_component_and_full463_audit' if not errors else 'terminal_evidence_with_audit_errors'
    result['limitations'] = [
        'HOST reexecutes saved integer Native verdicts and metadata/counter/source/clock arithmetic, not hidden/logit/KV tensors or finite predicates. Actual current sameM/FTF/64queryKV/compaction/next gates are separately bound by original audit189.',
        'SLO admission1 changes server-side pending order; all463 arrivals/output256/no-drop/clocks remain original. It does not prove an infinite-stream fairness bound or guarantee tight-service/performance gain.',
        'Cross-M all80 compatibility is bounded to the actual current component. A numerical-variant full remains a variant and is not silently adopted.',
        'Page highwater records HOST referenced544-block allocation for this trajectory only, not HBM bytes or arbitrary active-capacity safety.',
        'Native charged decode, outer generate wall and worker submission clocks are kept separate. Role-local HOST phases are not additive device costs or external-client measurements.',
        'Visible bonus subtraction is executed-trajectory accounting, not a rerun without bonus. No whole463 independent AR equality, production adoption or goal fulfillment is issued. A failed job remains failed.'
    ]
    result['UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(**desc(args.output), status=result['status'], job_status=result['job_status'], errors=errors)))
    return int(bool(errors))


if __name__ == '__main__': raise SystemExit(main())
