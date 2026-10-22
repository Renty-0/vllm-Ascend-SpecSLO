"""SOURCE ONLY: future frozen193 terminal evidence audit; stdlib, no backend.

Execution requires a separately authorized terminal receipt SHA. This file does
not launch work, acquire the device lock, or grant model/operator qualification.
Retained integer values are recomputed; hidden/KV tensors are not reconstructed.
"""
import argparse
import ast
import collections
import copy
import datetime
import gzip
import hashlib
import importlib.abc
import json
import math
import re
import statistics
import sys
import tarfile
import types
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-causal80-fused-bonus-verdict-full-v193'
BASE = B / 'tree-high-normal-cap4-native-families-v148'
NATIVE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
ACTION = 'causal80-fused-bonus-verdict-active32-full463'
FREEZE_SHA = '04eb943ebaf3c21fc0a855f4d0f1b72238c0c177a8da0b60e0ee3c4a1138a68a'
ROOT_REVIEW_SHA = '68594bbc7d31fd1eb4d5a81186e5871c7ecaadf40095471c9c0aa9f0421342a7'
HELPER = D / 'audit_v192_terminal_HOST_revision4.py'
HELPER_SHA = 'ac7af03ff58e8deeb3cca57c9811674d69e27d54c9ac96298893f5c8188f85b7'
CLOCK = D / 'analyze_completed_tree_full463_cost.py'
CLOCK_SHA = 'd3266647799fc0cf00c49fe98e195ecb85dc446e51ab66c643a5b1d5c3ca0b74'
BACKENDS = ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'transformers', 'safetensors', 'acl')

# Reuse only the exact already-reviewed generic stdlib helpers. No192 globals,
# candidate algorithm, receipt path or actual input is imported or evaluated.
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != HELPER_SHA:
    raise RuntimeError('Pinned192 revision4 generic HOST helper source changed')
_names = {'need', 'sha', 'read', 'desc', 'merged', 'Firewall', 'function', 'verify_archive', 'Values'}
_nodes = [copy.deepcopy(n) for n in ast.parse(HELPER.read_text()).body
          if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in _names]
if {n.name for n in _nodes} != _names:
    raise RuntimeError('Complete selected stdlib helper AST required')
exec(compile(ast.fix_missing_locations(ast.Module(body=_nodes, type_ignores=[])),
             str(HELPER) + '::isolated193-generic-HOST-only', 'exec'), globals())


def selected(freeze, root, epoch):
    cpu = root / f'CPU_RESULT{epoch}.json'
    return merged(*(freeze.get(k, {}) for k in ('files', 'external_files', 'retained_HOST_evidence',
                                               'execution_guard_inputs', 'peer_reviews')),
                  {str(cpu): freeze['CPU_sha256'], str(root / f'FROZEN_SOURCE{epoch}.json'):
                   sha(root / f'FROZEN_SOURCE{epoch}.json'),
                   freeze['root_review']['path']: freeze['root_review']['sha256']})


def source_evidence(receipt):
    fp = P / 'FROZEN_SOURCE193.json'
    need(sha(fp) == FREEZE_SHA, 'exact frozen193 source identity')
    f = read(fp)
    rp = Path(f['root_review']['path'])
    need(sha(rp) == f['root_review']['sha256'] == ROOT_REVIEW_SHA, 'exact ROOT193 review')
    pins = selected(f, P, 193)
    reviewed = {k: v for k, v in pins.items() if k not in (str(fp), str(rp))}
    root = read(rp)
    need(root['status'] == 'passed_source_review' and root['candidate'] == str(P) and
         root['reviewed_inputs'] == reviewed, 'ROOT exact reviewed input/CPU/peer/guard map')
    need(all(sha(path) == digest for path, digest in pins.items()), 'all selected current bytes')
    cpu = read(P / 'CPU_RESULT193.json')
    need(cpu['schema'] == 'V193.actual-fused-BONUS-source-HOST.v1' and cpu['status'] == 'passed'
         and cpu['tests_run'] == 4 and cpu['failures'] == cpu['errors'] == 0
         and cpu['Torch_imported'] is cpu['NPU_used'] is cpu['model_constructed'] is False
         and cpu['local_before'] == cpu['local_after'] == f['files']
         and cpu['external_before'] == cpu['external_after'] == f['external_files'],
         'actual193 CPU field names and current source pre/post; not192 schema')
    need(receipt['readiness']['freeze_sha256'] == FREEZE_SHA and
         receipt['readiness']['CPU_sha256'] == f['CPU_sha256'], 'actual spawned193 identity')
    ar = receipt['readiness']['archive']
    need(Path(ar['path']) == P / 'PREPARED_SOURCE_ARCHIVE193.tar.gz' and
         ar['members'] == len(pins) and sha(ar['path']) == ar['sha256'], 'own exact source archive')
    archives = {'P193': verify_archive(ar['path'], pins)}
    inputs = read(P / 'INPUTS193.json')
    descriptor = Path(inputs['parent_archive']['descriptor'])
    need(sha(descriptor) == inputs['parent_archive']['descriptor_sha256'], 'P148 descriptor pin')
    original = read(descriptor)
    need(sha(original['path']) == original['sha256'] == inputs['parent_archive']['sha256'],
         'original P148 binary SHA')
    archives['P148'] = verify_archive(original['path'], {'/' + n: h for n, h in original['members'].items()})
    for epoch, name in ((160, 'tree-active-window32-16-local-v160'), (166, 'tree-active-window-full-v166'),
                        (173, 'tree-causal80-real16-bankfix-full-v173')):
        parent = B / name
        pf = read(parent / f'FROZEN_SOURCE{epoch}.json')
        pin = inputs[f'P{epoch}_archive'] if epoch != 173 else inputs['parent173']['archive']
        need(sha(pin['path']) == pin['sha256'], f'unchanged P{epoch} parent archive binary')
        parent_pins = selected(pf, parent, epoch)
        need(all(sha(path) == digest for path, digest in parent_pins.items()), f'P{epoch} current selected bytes')
        archives[f'P{epoch}'] = verify_archive(pin['path'], parent_pins)
    need(sha(CLOCK) == CLOCK_SHA, 'independent unchanged Native/stream metric helper')
    for name in ('pre_verifier', 'post_verifier'):
        v = receipt[name]
        actual = read(v['path'])
        need(sha(v['path']) == v['sha256'] and v['exit_code'] == 0 and actual == v['actual_result']
             and actual['status'] == 'pass' and actual['errors'] == []
             and (actual['checked_frozen_files'], actual['checked_external_files'],
                  actual['checked_packages']) == (10496, 15, 212), 'saved original verifier ' + name)
    for name in ('devices_before', 'devices_after'):
        v = receipt[name]
        need(v['all_idle'] is v['four_910B2'] is True and set(v['processes']) == {'2', '5', '6', '7'}
             and all(z['exit_code'] == 0 and z['idle'] is True and z['output'].strip() ==
                     'No process in device.' for z in v['processes'].values()), 'saved four idle ' + name)
    need(receipt['sources_unchanged'] is True and receipt['owned_remaining'] == []
         and receipt['execution_lock']['nonblocking_exclusive'] is True and receipt['limits'] ==
         dict(child_execution_seconds=7200, actual_tag_bytes=256 * 1024 ** 2,
              aggregate_private_bytes=512 * 1024 ** 2), 'source/selflock/limits/owned cleanup')
    return dict(freeze=desc(fp), ROOT_review=desc(rp), CPU=desc(P / 'CPU_RESULT193.json'),
                selected_current_SHA_checked=len(pins), archives=archives,
                original_prepost='saved10496/15/212 PASS; not rerun', saved_all4_idle=True,
                inherited_device_qualification=False, original_HOST_suites_rerun=False)


def native_values(value, counters, failures, *, zero):
    need(set(value) == {'before', 'after', 'delta'} and
         all(set(value[k]) == set(counters) for k in value), 'exact Native21 snapshots')
    for key in counters:
        a, b, delta = (value[k][key] for k in ('before', 'after', 'delta'))
        need(type(a) is type(b) is type(delta) is int and 0 <= a <= b and b - a == delta,
             'strict Native integer subtraction ' + key)
        need((not zero or delta == 0) and (key not in failures or delta == 0), 'actual Native zero/failure gate ' + key)
    return len(counters) if zero else 0


def execution(value):
    # NativeGraphExecution dataclass serializes these FOUR fields. used_aclgraph
    # is a property, never an asdict member. Do not repeat the192 schema mistake.
    need(type(value) is dict and set(value) == {'mode', 'fallback_reason', 'capture_attempted', 'replay_executed'}
         and value['mode'] in ('capture_replay', 'replay') and value['fallback_reason'] is None
         and type(value['capture_attempted']) is type(value['replay_executed']) is bool
         and value['replay_executed'] is True
         and value['capture_attempted'] is (value['mode'] == 'capture_replay'),
         'actual returned Native graph execution, not a config flag/property guess')


def bonus(rows, targets):
    """Independent complete BONUS integer algebra; no tensor/kernel execution."""
    need(len(rows) == 16 and len(targets) == 80 and all(len(row) == 4 for row in rows), 'N16 root+4 windows')
    tokens, paths, pairs = [], [], []
    for i, row in enumerate(rows):
        head = targets[5 * i:5 * i + 5]
        need(all(type(v) is int and v >= 0 for v in [*row, *head]), 'actual nonnegative token IDs')
        accepted = next((j for j in range(4) if row[j] != head[j]), 4)
        pairs.append([accepted, head[accepted]])
        tokens.append(row[:accepted] + [head[accepted]] + [-1] * (4 - accepted))
        paths.append(list(range(accepted)) + [-1] * (4 - accepted))
    return tokens, paths, pairs


def component_extra(directory, reports):
    from scope_v146 import NATIVE_COUNTERS, FAILURES
    host = types.SimpleNamespace(Tensor=Values, tensor=lambda values, **kw: Values(values), long='long')
    namespace = dict(torch=host, TreeVerificationOutput=lambda a, b:
                     types.SimpleNamespace(token_ids=a, accepted_node_indices=b))
    verify = function(NATIVE / 'native_engine.py', '_verify_tree_outputs_host', namespace)
    plans = [types.SimpleNamespace(candidate_budget=4, parent_indices=Values([-1, 0, 1, 2])) for _ in range(16)]
    folder = directory / 'causal80-own-current-process'
    component = [read(folder / f'rank{rank}-component.json.gz') for rank in range(4)]
    records = reports[1]['component']['fused_bonus']['operator_records']
    need([(r['pass_index'], r['cycle']) for r in records] == [(p, c) for p in range(3) for c in range(4)],
         'twelve actual current model records in exact order')
    zeros = 0
    compared = 0
    all4_commits = 0
    graph_executions = 0
    for rank in range(4):
        need(len(component[rank]['traces']) == 3, 'three current complete fresh histories')
        for phase, trace in enumerate(component[rank]['traces']):
            zeros += native_values(trace['Native'], NATIVE_COUNTERS, FAILURES, zero=phase > 0)
        evidence = reports[rank]['component']['fused_bonus']
        need(evidence['qualified'] is True and evidence['new_model_calls'] == evidence['new_Native_keys'] == 0
             and evidence['evidence'] == reports[1]['component']['fused_bonus']['evidence']
             and evidence['operator_records'] == (records if rank == 1 else []), 'current leader-only BONUS and all4 issuer evidence')
        for phase in range(3):
            for cycle in range(4):
                row = read(folder / f'rank{rank}-pass{phase}-cycle{cycle}.json.gz')
                commit = row['commit']
                need(all(type(commit[k]) in (int, bool) and commit[k] == 0 for k in
                         ('actual_structural_failed', 'actual_numerical_failed'))
                     and commit['original_finish_cycle_executed'] is True
                     and row['independent_commit']['histories_equal'] is True
                     and row['independent_commit']['candidate_KV_copied'] is False, 'original all4 commit and independent history')
                leader = read(folder / f'rank1-pass{phase}-cycle{cycle}.json.gz')
                common = ('cycle_id', 'target_home', 'committed_tokens', 'accepted_paths',
                          'actual_prefix_epochs', 'actual_logical_tokens')
                need(all(commit[k] == leader['commit'][k] for k in common) and row['rows'] == leader['rows'],
                     'same four-rank normal verdict/cycle; role-local counters not assumed equal')
                all4_commits += 1
                if rank == 0:
                    need(row['target'] is row['reference'] is row['fused_bonus_check'] is None,
                         'draft role cannot claim target tensor comparison')
                else:
                    head = row['reference']['real_query_tokens']
                    tokens, paths, pairs = bonus(row['rows'], head)
                    original = verify(row['rows'], plans, Values(head), 4, 'explicit-HOST-integer-values')
                    need(original.token_ids.values == tokens and original.accepted_node_indices.values == paths,
                         'independent BONUS matches exact Native verifier including two-dimensional packed slicing')
                    need([[t for t in value if t >= 0] for value in tokens] == commit['committed_tokens']
                         and paths == commit['accepted_paths'] and head == leader['reference']['real_query_tokens'],
                         'all three target heads and actual all4 accepted/committed values')
                    check = records[phase * 4 + cycle]
                    need(check['model_tokens'] == tokens and check['model_paths'] == paths
                         and check['independent_sameM_model_equal'] is check['current_target_input_checked'] is
                         check['pinned_copy_event_consumed'] is True and leader['fused_bonus_check'] == check,
                         'actual consumed fused result independently recomputed from retained full80 head')
                    for family in ('greedy', 'hidden'):
                        need([x['phase'] for x in row['target'][family]] == ['first', 'changed', 'restored'], 'actual F/T/F sequence')
                        for item in row['target'][family]:
                            execution(item['execution']); graph_executions += 1
                    compared += 1
        # The producer generates nextdraft records on all roles, but numerical
        # draft proof belongs ONLY to rank0; target roles have no draft proof.
        for phase in range(3):
            nxt = read(folder / f'rank{rank}-pass{phase}-next.json.gz')
            need({k: nxt[k] for k in ('source', 'selected', 'rows', 'confidences')} ==
                 component[rank]['traces'][phase]['next_draft'], 'actual nextdraft values, no invented target-role model execution')
            need(type(nxt['draft']) is dict if rank == 0 else nxt['draft'] is None, 'rank0-only nextdraft numerical proof')
    controlled = records[0]['controlled_operator']
    tokens, paths, pairs = bonus(controlled['drafts'], [v for row in controlled['targets'] for v in row])
    original = verify(controlled['drafts'], plans, Values([v for row in controlled['targets'] for v in row]), 4, 'HOST-values')
    need(pairs == controlled['pairs'] and [v[0] for v in pairs] == controlled['expected_accept'] == [i % 5 for i in range(16)]
         and original.token_ids.values == tokens and original.accepted_node_indices.values == paths
         and controlled['actual_original_NPU_operator'] is controlled['bonus_enabled_all16'] is True,
         'retained real operator mismatch0..3/all4 accept/bonus/zero/EOS pairs recomputed')
    need(all(records[i]['controlled_operator'] is None for i in range(1, 12)) and zeros == 168
         and compared == 36 and all4_commits == 48 and graph_executions == 216, 'bounded actual component counts')
    return dict(complete_all4_commits=all4_commits, actual_target_integer_verifier_calls=compared,
                target_owner_verdicts=compared * 16, controlled_operator_pairs=16,
                model_consumed_results_recomputed=12, actual_four_field_graph_executions=graph_executions,
                component_two_zero_Native_subtractions=zeros, hidden_KV_recomputed_on_HOST=False,
                model_compact_pairs_retained=False, model_full80_heads_and_consumed_tokens_retained=True,
                controlled_compact_pairs_retained=True, candidate_KV_used_as_oracle=False)


def full_extra(directory, reports):
    from scope_v146 import NATIVE_COUNTERS, FAILURES
    checks = []
    full = None
    zeros = 0
    for index, role in enumerate(('warmup65', 'zero65-1', 'zero65-2', 'full463')):
        traces = [read(directory / f'windows/window32/{role}/rank{rank}-{role}-actual-trace.json.gz') for rank in range(4)]
        calls = traces[1]['normal_target_calls']
        need(all(t['normal_target_calls'] == calls for t in traces[1:]) and traces[0]['normal_target_calls'] == [], 'actual target call leader/TP schema')
        b5 = sum(5 in c['budgets'] for c in calls)
        heterogeneous = sum(len(set(c['budgets'])) > 1 for c in calls)
        rank_counts = []
        for rank, trace in enumerate(traces):
            zeros += native_values(trace['native'], NATIVE_COUNTERS, FAILURES, zero=index > 0)
            entry = reports[rank]['windows']['traces'][index]
            need(entry['role'] == role and entry['window'] == 32, 'actual pair160 worker envelope ordering')
            raw = entry['original_trace']['fused_bonus_counts']
            need(raw['pending_empty'] is True and set(raw['delta']) == set(raw['after']) | set(raw['before']), 'explicit complete verdict counter field union')
            delta = {}
            for key in set(raw['after']) | set(raw['before']):
                a, z = raw['before'].get(key, 0), raw['after'].get(key, 0)
                need(type(a) is type(z) is type(raw['delta'][key]) is int and 0 <= a <= z and raw['delta'][key] == z - a,
                     'actual integer verdict counter subtraction')
                delta[key] = z - a
            need(delta.get('queue_failed', 0) == delta.get('finish_failed', 0) == 0 and delta.get('packet', 0) == len(calls),
                 'same one successful status packet per actual target call on every rank')
            if rank == 1:
                need(delta.get('queued', 0) == delta.get('consumed', 0)
                     and delta.get('queued', 0) + delta.get('fallback', 0) == len(calls)
                     and delta.get('fallback_B5', 0) == b5 and delta.get('fallback_heterogeneous', 0) == heterogeneous,
                     'actual fast/fallback/call and B5/heterogeneous category correspondence')
                if role == 'full463':
                    need(delta.get('consumed', 0) > 0 and b5 > 0 and heterogeneous > 0,
                         'this current full actually exercises BONUS and both generic fallback classes')
            else:
                need(delta.get('queued', 0) == delta.get('consumed', 0) == 0, 'only actual target leader stages/consumes')
            rank_counts.append(delta)
        checks.append(dict(role=role, actual_target_calls=len(calls), B5_calls=b5, heterogeneous_calls=heterogeneous,
                           categories_overlap_not_a_partition=True, counts_by_rank=rank_counts))
        if role == 'full463': full = traces
    need(zeros == 252, 'four ranks times twozero+full times21 counters; includes168 twozero and84 full')
    trace = full[1]
    namespace = dict(math=math, statistics=statistics, CLASSES=('tight', 'normal', 'loose'))
    function(CLOCK, 'need', namespace)
    metrics = function(CLOCK, 'clocks', namespace)(trace)
    saved = reports[1]['windows']['per_rank_full_metrics'][0]['output']
    need(all(saved[k] == v for k, v in metrics.items()), 'independent463x256 stream/Native-clock metric arithmetic')
    first = {}
    for event in trace['stream_events']:
        i = event['request_index']
        if i not in first:
            need('first_token_wall_seconds160' in event, 'actual first callback wall sample')
            first[i] = event['first_token_wall_seconds160']
        else: need('first_token_wall_seconds160' not in event, 'exactly one first callback sample')
    need(set(first) == set(range(463)), 'all463 first-token samples')
    ttft = [(first[i] - row['arrival_ts']) * 1000 for i, row in enumerate(trace['generate_results'])]
    need(all(math.isfinite(v) and v >= 0 for v in ttft) and all(row['arrival_ts'] == trace['absolute_arrivals'][i]
         for i, row in enumerate(trace['generate_results'])), 'original arrival and nonnegative first-token wall')
    p = sorted(ttft); at = (len(p) - 1) * .9
    median, p90 = statistics.median(p), p[math.floor(at)] + (p[math.ceil(at)] - p[math.floor(at)]) * (at - math.floor(at))
    need(saved['TTFT160']['median_ms'] == median and saved['TTFT160']['P90_ms'] == p90, 'retained actual TTFT percentiles')
    for rank, value in enumerate(full):
        need(value['sealed_before'] is value['sealed_after'] is value['cache_released'] is True and
             (value['generate_results'] is None if rank != 1 else len(value['generate_results']) == 463),
             'all4 sealed/released; output owned only by rank1')
    calls = trace['normal_target_calls']
    shapes = collections.Counter((c['physical_M'], c['physical_S'], c['family']) for c in calls)
    return dict(metrics=metrics, TTFT_worker_samples=dict(samples=463, median_ms=median, P90_ms=p90),
                call_counter_checks=checks, twozero21_subtractions=168, sealed_full21_subtractions=84,
                actual_target_calls=len(calls), owner_rounds=sum(c['N'] for c in calls),
                output_tokens=463 * 256, request_output_sha256=[hashlib.sha256(json.dumps(row['completion_token_ids'],
                separators=(',', ':')).encode()).hexdigest() for row in trace['generate_results']],
                geometry=[dict(M=m, S=s, family=f, calls=n) for (m, s, f), n in sorted(shapes.items())],
                performance_observation_only=True, whole_workload_AR_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terminal-authorized', action='store_true', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--expected-receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.tag) is not None, 'safe exact terminal tag')
    need(re.fullmatch('[0-9a-f]{64}', args.expected_receipt_sha256) is not None, 'explicit receipt SHA')
    need(args.output.parent.resolve() == D and not args.output.exists(), 'fresh private D-only audit output')
    rp = P / f'receipt-{args.tag}.json'
    need(not rp.is_symlink() and sha(rp) == args.expected_receipt_sha256, 'ROOT-authorized exact terminal193 receipt')
    receipt = read(rp)
    need(receipt['status'] in ('complete', 'failed') and receipt['action'] == ACTION and receipt['tag'] == args.tag,
         'only actual terminal193 producer schema; never read running output')
    directory = P / 'actual' / args.tag
    need(directory.is_dir() and directory.resolve() == directory and not directory.is_symlink(), 'own terminal actual directory')
    sys.meta_path.insert(0, Firewall())
    errors = []
    result = dict(schema='V193.independent-terminal-HOST-audit.v1', candidate=str(P), tag=args.tag,
                  job_status=receipt['status'], receipt=desc(rp), script=desc(__file__), errors=errors,
                  Torch_imported=False, NPU_used=False, model_run=False, formal_Goodput=False, adopted=False)
    def attempt(label, callback):
        try:
            result[label] = callback()
        except Exception as error:
            errors.append(dict(stage=label, error=repr(error)))
    attempt('provenance', lambda: source_evidence(receipt))
    def terminal():
        need(read(directory / 'worker-exit.json') == dict(closed=True, all_exited=True), 'actual worker process pool closed')
        path = P / f'local-{args.tag}.log'
        need(Path(receipt['child']['log']) == path and not path.is_symlink(), 'own actual child log')
        return dict(worker_exit=desc(directory / 'worker-exit.json'), child_log=desc(path),
                    worker_errors=[dict(file=desc(p), error=read(p)) for p in sorted(directory.glob('rank*-last-error.json'))])
    attempt('terminal_closure', terminal)
    def manifest():
        paths = sorted(directory.rglob('*'))
        need(not any(p.is_symlink() for p in paths), 'no actual raw symlink aliases')
        values = {str(p): desc(p) for p in paths if p.is_file()}
        out = args.output.with_name(args.output.stem + '_RAW_MANIFEST.json')
        need(not out.exists(), 'fresh independent raw manifest')
        with out.open('x') as stream: json.dump(values, stream, indent=2, sort_keys=True); stream.write('\n')
        return dict(**desc(out), files=len(values), bytes_total=sum(v['bytes'] for v in values.values()))
    attempt('raw_manifest', manifest)
    if 'provenance' in result:
        sys.path.insert(0, str(P))
        def original():
            import audit193
            actual = audit193.audit(directory, (32,))
            completed = read(directory / 'completed.json')
            need(actual['passed'] is True and actual == completed['audit'] == receipt['actual_full_audit']
                 and completed['action'] == ACTION and completed['replies'] ==
                 [['calibrated', i, dict(windows=[32], full463=True)] for i in range(4)], 'unchanged audit equals actual runtime/supervisor')
            need(receipt['status'] == 'complete' and receipt['child']['exit_code'] == 0 and receipt['errors'] == []
                 and receipt['actual_execution_complete'] is True and receipt['actual_full_status'] ==
                 ('passed_full' if actual['old_padding_behavior_equal'] else 'passed_numerical_variant_full'),
                 'actual whole job success separate from any retained inference')
            return dict(source=desc(P / 'audit193.py'), result=actual, reused_current_runtime_gates=True,
                        independent_tensor_recomputation=False)
        attempt('original_unmodified_audit', original)
        def additional():
            reports = [read(directory / f'rank{r}-local-diagnostic.json') for r in range(4)]
            return dict(component_integer_recalculation=component_extra(directory, reports), full463=full_extra(directory, reports))
        attempt('independent_retained_integer_and_full_arithmetic', additional)
    need(not Firewall.attempts and not any(n.split('.')[0] in BACKENDS for n in sys.modules), 'no backend imports')
    result['status'] = ('passed_terminal_component_and_full463_audit' if not errors else
                        'terminal_evidence_failed_or_incomplete')
    result['first_audit_failure'] = errors[0] if errors else None
    result['original_terminal_errors'] = receipt.get('errors', [])
    result['limitations'] = [
        'Integer BONUS/Native verifier arithmetic is recomputed from retained current component full80 heads, proposals and consumed results. Only the controlled operator fixture retains raw compact pairs; the twelve model checks do not retain their pinned compact-pair tensor.',
        'Full463 does not retain per-call target logits/heads, pinned staging tensors or status-packet payloads. Its actual consume/fallback/packet success counts and retained generic-call categories are audited against the pinned universal packet implementation, not reconstructed as full per-call numerical or packet-byte equality.',
        'The exact original current same-M physical hidden/head/all128 target KV, rank0 draft all56 KV, commit/compaction/next, natural F/T/F and finite/protocol gates are bound through original evidence. No hidden/KV tensors are numerically revalidated on HOST.',
        'NativeGraphExecution is checked using its four actual serialized fields; used_aclgraph is a property. Source-only HOST evidence and old device receipts never grant current operator/graph/model qualification.',
        'The unchanged original463 prompts, arrivals,256 output tokens, callbacks and original clocks are audited. No independent whole-workload AR equality or losslessness claim is issued.',
        'Native/stream timing groups and HOST phases include overlap and waits, and are not added or relabeled pure device time. Performance remains observational; no adoption, speed guarantee or goal completion.',
        'A failed terminal job remains failed, even if retained component/full artifacts can be audited. Missing files become explicit incomplete stages, not a substituted passing record.'
    ]
    result['UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(**desc(args.output), status=result['status'], job_status=result['job_status'], errors=errors)))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
