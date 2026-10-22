"""Independent frozen171/177 terminal evidence audit; no model/device work.

Generic stdlib helpers are reused by exact AST from the completed173 auditor.
Actual171 is opened only with an explicit terminal receipt SHA/authorization.
"""
import argparse, ast, collections, copy, datetime, gzip, hashlib, importlib.abc
import json, math, re, statistics, sys, tarfile, types
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-normal128-eager64-global192-full-v171'
OVERLAY = D / 'v177-root-safe-storage-overlay'
ACTION = 'normal128-eager64-global192-full463'
HELPERS = D / 'audit_v173_terminal_full463_HOST.py'
HELPER_SHA = 'ed8aef51646ccf60c4812782d080d119fa406d7e53138ce24d6b470bce74c6f5'
CLOCKS = D / 'analyze_completed_tree_full463_cost.py'
CLOCK_SHA = 'd3266647799fc0cf00c49fe98e195ecb85dc446e51ab66c643a5b1d5c3ca0b74'
FREEZE_SHA = 'ef0538fe4d0647045e57fa40edc7cc151bc3c0bc626c404373fa6da9f950a0f5'
OVERLAY_FREEZE_SHA = '7787e359fdf8f8841c14bba1ebf648327a4eec60bee022c8d3deccc746295a68'
NATIVE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
BACKENDS = ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'transformers', 'safetensors', 'acl')

if hashlib.sha256(HELPERS.read_bytes()).hexdigest() != HELPER_SHA:
    raise RuntimeError('Exact completed173 HOST helper source changed')
_names = {'need', 'sha', 'desc', 'read', 'archive', 'Firewall', 'extracted', 'Value', 'pct'}
_nodes = [copy.deepcopy(n) for n in ast.parse(HELPERS.read_text()).body
          if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in _names]
if {n.name for n in _nodes} != _names:
    raise RuntimeError('Complete selected stdlib173 helper AST')
exec(compile(ast.fix_missing_locations(ast.Module(body=_nodes, type_ignores=[])),
             str(HELPERS) + '::isolated171-stdlib-helpers', 'exec'), globals())


def provenance(receipt, tag):
    freeze_path = P / 'FROZEN_SOURCE171.json'
    need(sha(freeze_path) == FREEZE_SHA, 'unchanged frozen171 identity')
    f = read(freeze_path)
    pins = {}
    for k in ('files', 'external_files', 'retained_HOST_evidence', 'execution_guard_inputs', 'peer_reviews'):
        for path, digest in f[k].items():
            need(path not in pins or pins[path] == digest, 'consistent selected source ' + path)
            pins[path] = digest
    pins.update({str(P / 'CPU_RESULT171.json'): f['CPU_sha256'], str(freeze_path): FREEZE_SHA,
                 f['root_review']['path']: f['root_review']['sha256']})
    need(len(pins) == 140 and all(sha(path) == digest for path, digest in pins.items()),
         'actual140 selected frozen bytes')
    need(sha(CLOCKS) == CLOCK_SHA, 'pinned independent clocks helper')
    need(receipt['readiness']['freeze_sha256'] == FREEZE_SHA and
         receipt['readiness']['CPU_sha256'] == f['CPU_sha256'], 'actual171 freeze/CPU')
    ar = receipt['readiness']['archive']
    need(Path(ar['path']) == P / 'PREPARED_SOURCE_ARCHIVE171.tar.gz' and
         ar['members'] == 140 and sha(ar['path']) == ar['sha256'], 'actual140 archive identity')
    own_archive = archive(ar['path'], pins)
    # This is a separately frozen storage-only overlay, not an altered model epoch.
    of_path = OVERLAY / 'FROZEN_OVERLAY177.json'
    need(sha(of_path) == OVERLAY_FREEZE_SHA, 'frozen177 overlay identity')
    of = read(of_path)
    opins = dict(of['inputs']); opins[str(of_path)] = OVERLAY_FREEZE_SHA
    need(all(sha(path) == digest for path, digest in opins.items()), 'all frozen177 source bytes')
    oa = archive(OVERLAY / 'PREPARED_OVERLAY177.tar.gz', opins)
    registry = read(OVERLAY / 'REGISTRY177.json')
    entry = next(v for v in registry['entries'] if v['epoch'] == 171)
    need(entry['candidate'] == str(P) and entry['action'] == ACTION and
         entry['freeze_sha256'] == FREEZE_SHA and sha(entry['guard']) == entry['guard_sha256'],
         'already registered exact171 original guard')
    ov_path = OVERLAY / ('receipt-overlay-171-' + tag + '.json')
    ov = read(ov_path)
    need(ov['original_receipt_sha256'] == sha(P / ('receipt-' + tag + '.json')) and
         ov['original_status'] == receipt['status'] and ov['runtime_or_model_source_changed'] is False
         and ov['outer_lock_added'] is False and ov['source_before'] == ov['source_after'] ==
         dict(freeze_sha256=OVERLAY_FREEZE_SHA, archive_sha256=oa['sha256'], inputs=len(opins)),
         'separate exact storage overlay receipt/source closure')
    if receipt['status'] == 'complete':
        need(ov['exit_code'] == 0 and ov['errors'] == [], 'successful overlay wrapper')
    for k in ('pre_verifier', 'post_verifier'):
        v = receipt[k]; q = read(v['path'])
        need(sha(v['path']) == v['sha256'] and v['exit_code'] == 0 and q == v['actual_result']
             and q['status'] == 'pass' and q['errors'] == [] and
             (q['checked_frozen_files'], q['checked_external_files'], q['checked_packages']) ==
             (10496, 15, 212), 'saved original source verifier ' + k)
    for k in ('devices_before', 'devices_after'):
        v = receipt[k]
        need(v['all_idle'] is v['four_910B2'] is True and set(v['processes']) == {'2', '5', '6', '7'} and
             all(x['exit_code'] == 0 and x['idle'] is True and x['output'].strip() == 'No process in device.'
                 for x in v['processes'].values()), 'saved four idle ' + k)
    need(receipt['sources_unchanged'] is True and receipt['owned_remaining'] == [] and
         receipt['execution_lock']['nonblocking_exclusive'] is True and receipt['limits'] ==
         dict(child_execution_seconds=7200, actual_tag_bytes=256 * 1024 ** 2,
              aggregate_private_bytes=512 * 1024 ** 2), 'original selflock/limits/source/owned cleanup')
    return dict(freeze=desc(freeze_path), CPU=desc(P / 'CPU_RESULT171.json'), selected_SHA_checked=140,
                archive=own_archive, overlay=dict(receipt=desc(ov_path), freeze=desc(of_path), archive=oa,
                inputs=len(opins), storage_only=True, outer_lock_added=False),
                saved_original_prepost='10496/15/212 PASS; not rerun', saved_all4_idle=True,
                owned_remaining=[], complete_environment_import_closure_claimed=False)


def component_integer_verdicts(directory):
    """Reexecute the unchanged Native integer verifier, not tensor mathematics."""
    from mathscope171 import cases
    ht = types.SimpleNamespace(Tensor=Value, tensor=lambda v, **kw: Value(v), long='long')
    ns = dict(torch=ht, TreeVerificationOutput=lambda a, b:
              types.SimpleNamespace(token_ids=a, accepted_node_indices=b))
    verify = extracted(NATIVE / 'native_engine.py', '_verify_tree_outputs_host', ns)
    calls = owners = zeros = 0
    folder = directory / 'normal128-own-full-process'
    for rank in range(4):
        report = read(folder / f'rank{rank}-component.json.gz')
        need(len(report['traces']) == 6, 'two cases times3 fresh histories')
        for trace in report['traces']:
            before, after, delta = (trace['Native'][k] for k in ('before', 'after', 'delta'))
            need(len(delta) == 21 and set(before) == set(after) == set(delta) and
                 all(type(before[k]) is type(after[k]) is type(delta[k]) is int and
                     0 <= before[k] <= after[k] and after[k] - before[k] == delta[k] for k in delta),
                 'independent current165 component counter arithmetic')
            if trace['zero_replay']:
                need(not any(delta.values()), 'complete fresh-history component zero')
                zeros += len(delta)
        if not rank:
            continue
        for phase in range(3):
            for case in cases():
                for cycle in range(4):
                    row = read(folder / f'rank{rank}-pass{phase}-case{case["index"]}-cycle{cycle}.json.gz')
                    plans = [types.SimpleNamespace(candidate_budget=v, parent_indices=Value([-1, 0, 1, 2][:v]))
                             for v in case['budgets']]
                    out = verify(row['rows'], plans, Value(row['reference']['real_query_tokens']), 4, 'HOST-values')
                    need([[t for t in r if t >= 0] for r in out.token_ids.values] ==
                         row['commit']['committed_tokens'] and out.accepted_node_indices.values ==
                         row['commit']['accepted_paths'], 'actual C97/C128 head vs unchanged integer verifier/commit')
                    calls += 1; owners += 32
    need(zeros == 4 * 2 * 2 * 21 and calls == 72 and owners == 2304, 'all336zero/72 integer comparisons')
    return dict(actual_current_integer_verifier_calls=calls, owner_verdicts=owners,
                raw_component_zero21_subtractions=zeros, hidden_KV_recomputed_on_HOST=False)


def full_statistics(traces):
    t = traces[1]
    ns = dict(math=math, statistics=statistics, CLASSES=('tight', 'normal', 'loose'))
    extracted(CLOCKS, 'need', ns)
    metrics = extracted(CLOCKS, 'clocks', ns)(t)
    first = {}
    for e in t['stream_events']:
        owner = e['request_index']
        if owner not in first:
            need('first_token_wall_seconds160' in e, 'worker first-submission sample')
            first[owner] = e['first_token_wall_seconds160']
        else:
            need('first_token_wall_seconds160' not in e, 'one worker first sample')
    need(set(first) == set(range(463)), 'all463 worker TTFT samples')
    ttft = [(first[i] - r['arrival_ts']) * 1000 for i, r in enumerate(t['generate_results'])]
    need(all(math.isfinite(v) and v >= 0 for v in ttft), 'actual nonnegative worker TTFT')
    metrics['TTFT_worker_submission'] = dict(samples=463, **pct(ttft), by_class={
        cl: pct([ttft[i] for i, r in enumerate(t['generate_results']) if r['slo_class'] == cl])
        for cl in ('tight', 'normal', 'loose')})
    calls = t['normal_target_calls']; new = []; shapes = collections.Counter()
    for c in calls:
        n, q, C = c['N'], c['real_query_count'], sum(c['budgets'])
        shapes[(c['physical_M'], c['physical_S'], c['family'])] += 1
        if C > 96:
            s = 34 if n == 32 and q >= 158 else 36
            d = s - n; base, rem = divmod(165 - q, d)
            padding = [base + int(i < rem) for i in range(d)]
            need(25 <= n <= 32 and 97 <= C <= min(4 * n, 128) and q == C + n <= 160 and
                 c['family'] == 'causal' and all(2 <= x <= 9 for x in padding) and
                 (c['physical_M'], c['physical_S'], c['physical_query_partition']) ==
                 (165, s, [b + 1 for b in c['budgets']] + padding), 'actual current C97..128 dispatch')
            new.append(c)
    owner_rounds = sum(c['N'] for c in calls)
    queues = collections.defaultdict(collections.deque)
    for c in calls:
        for owner, budget in zip(c['sequence_ids'][:c['N']], c['budgets'], strict=True):
            queues[owner].append(budget)
    initial = set(); delivered = visible_bonus = event_rows = 0; align_error = None
    for e in t['stream_events']:
        owner = e['request_index']
        if owner not in initial:
            need(len(e['token_ids']) == 1, 'one initial prefill token')
            initial.add(owner); continue
        delivered += len(e['token_ids']); event_rows += 1
        if queues[owner]:
            budget = queues[owner].popleft()
            visible_bonus += int(len(e['token_ids']) == min(budget, 4) + 1)
        else:
            align_error = 'Post-initial worker callbacks include a row outside the retained verifier occurrences.'
    if any(queues.values()): align_error = 'Some verifier occurrences have no distinct post-initial callback.'
    dc = t['decode_counters']; accepted = dc['spec_rhythm_tree_accepted_nodes']; full = dc['spec_rhythm_tree_full_accepts']
    bonus = dict(owner_rounds=owner_rounds, accepted_nodes=accepted, Native_full_primary_accepts=full,
                 delivered_decode_tokens=delivered, verified_event_alignment_exact=align_error is None,
                 event_alignment_limitation=align_error)
    if align_error is None:
        discard = accepted + owner_rounds - delivered
        need(discard >= full - visible_bonus >= 0, 'primary bonus/output256 accounting')
        bonus.update(accepted_nodes_per_owner=accepted / owner_rounds,
                     uncapped_verifier_with_extra_target_per_owner=(accepted + owner_rounds) / owner_rounds,
                     delivered_after_limit_per_owner=delivered / owner_rounds,
                     visible_primary_full_bonus=visible_bonus, discarded_primary_bonus=full - visible_bonus,
                     total_output_limit_discarded=discard,
                     delivered_minus_visible_primary_bonus_per_owner=(delivered - visible_bonus) / owner_rounds,
                     subtraction_is_not_a_rerun_without_bonus=True)
    pages = {}
    for r in traces:
        a = r['allocator']
        need(a['pool_blocks'] == 544 and a['allocation_storage_rows'] == [527] and
             0 <= a['referenced_page_highwater'] <= 544 and a['final_referenced_pages'] == 0 and
             a['acquired_pages'] == a['released_pages'] and r['cache_released'] is True,
             'actual per-rank page highwater/release')
        pages[str(r['rank'])] = a
    return dict(metrics=metrics, clock_scope='Native charged decode and worker-submitted stream events',
                target_calls=len(calls), owner_rounds=owner_rounds, mean_N=owner_rounds / len(calls),
                N_histogram=dict(sorted(collections.Counter(c['N'] for c in calls).items())),
                N_gt16_calls=sum(c['N'] > 16 for c in calls), C_gt96_calls=len(new),
                actual_new165_domain_exercised=bool(new),
                new165_S_histogram=dict(sorted(collections.Counter(c['physical_S'] for c in new).items())),
                N32_Q160_calls=sum(c['N'] == 32 and c['real_query_count'] == 160 for c in new),
                shapes=[dict(M=m, S=s, family=f, calls=count) for (m, s, f), count in sorted(shapes.items())],
                bonus_accounting=bonus, allocator_by_rank=pages, configured=64, storage=527, pages=544,
                actual_active_peaks={str(r['rank']): r['active_peak'] for r in traces})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--terminal-authorized', action='store_true', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--expected-receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.tag) is not None, 'safe tag')
    need(args.output.parent.resolve() == D and not args.output.exists(), 'fresh D-only output')
    rp = P / ('receipt-' + args.tag + '.json')
    need(sha(rp) == args.expected_receipt_sha256, 'explicit terminal171 receipt SHA')
    r = read(rp); need(r['status'] in ('complete', 'failed') and r['action'] == ACTION,
                       'only exact terminal171 action may be inspected')
    A = P / 'actual' / args.tag
    need(A.is_dir() and A.resolve() == A and not A.is_symlink(), 'private terminal171 actual')
    sys.meta_path.insert(0, Firewall()); errors = []
    result = dict(schema='V171.independent-terminal-HOST-audit.v1', candidate=str(P), tag=args.tag,
                  job_status=r['status'], receipt=desc(rp), script=desc(__file__), errors=errors,
                  Torch_imported=False, NPU_used=False, adopted=False, formal_Goodput=False)
    log = P / ('local-' + args.tag + '.log')
    need(Path(r['child']['log']) == log and not log.is_symlink(), 'own exact terminal log')
    result['child_log'] = desc(log)
    try:
        result['provenance'] = provenance(r, args.tag)
        need(read(A / 'worker-exit.json') == dict(closed=True, all_exited=True), 'all workers closed')
    except Exception as error:
        errors.append(dict(stage='source_or_terminal_provenance', error=repr(error)))
    paths = sorted(A.rglob('*')); need(not any(p.is_symlink() for p in paths), 'no terminal raw alias')
    manifest = {str(p): desc(p) for p in paths if p.is_file()}
    mp = args.output.with_name(args.output.stem + '_RAW_MANIFEST.json'); need(not mp.exists(), 'fresh manifest')
    mp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    result['raw_manifest'] = dict(**desc(mp), files=len(manifest))
    result['worker_errors'] = [read(p) for p in sorted(A.glob('rank*-last-error.json'))]
    if 'provenance' in result:
        sys.path.insert(0, str(P))
        try:
            import audit171
            actual = audit171.audit(A, (64,)); completed = read(A / 'completed.json')
            need(actual['passed'] is True and actual == completed['audit'] == r['actual_full_audit'] and
                 completed['action'] == ACTION and completed['replies'] ==
                 [['calibrated', i, dict(windows=[64], full463=True)] for i in range(4)],
                 'unchanged audit171 equals completed/supervisor exact schema')
            result['original_unmodified_audit'] = dict(source=desc(P / 'audit171.py'), result=actual)
            result['additional_integer_component'] = component_integer_verdicts(A)
            need(r['status'] == 'complete' and r['child']['exit_code'] == 0 and r['errors'] == [] and
                 r['actual_execution_complete'] is True, 'job success separate from retained inference')
        except Exception as error:
            errors.append(dict(stage='original_component_full_audit_or_job_completion', error=repr(error)))
        try:
            traces = [read(A / 'windows/window64/full463' / f'rank{i}-full463-actual-trace.json.gz') for i in range(4)]
            result['retained_full463'] = full_statistics(traces)
        except Exception as error:
            errors.append(dict(stage='retained_full463_arithmetic', error=repr(error)))
    need(not Firewall.attempts and not any(n.split('.')[0] in BACKENDS for n in sys.modules), 'no backend imports')
    result['status'] = 'passed_terminal_component_and_full463_audit' if not errors else 'terminal_evidence_with_audit_errors'
    result['limitations'] = ['No hidden/head/KV tensors are recomputed from marker booleans on HOST. The original current sameM/all165/head/all64KV/FTF/compaction/own-next/twozero gates are separately bound by the unchanged original audit.',
        'C128/E64/global192 and active64 are joint changes. LowC retains the original mapper, not173 M80. This is not a single-variable comparison with best173.',
        'Page highwater is actual referenced HOST cache blocks, not HBM bytes; it proves only this completed trajectory. No page-aware admission or arbitrary active64 fit is inferred.',
        'Bonus subtraction is accounting on the executed trajectory, not a model rerun with bonus disabled. An unmatched callback/target-only residual is disclosed rather than invented.',
        'Timing groups are Native charged decode and worker-submitted events, not an external endpoint latency measurement. No HOST phase times are summed across roles or called additive device costs.',
        'No full-workload independent AR equality, guaranteed performance gain, production adoption or goal completion is issued. A failed terminal job remains failed.']
    result['UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(**desc(args.output), status=result['status'], job_status=result['job_status'], errors=errors)))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
