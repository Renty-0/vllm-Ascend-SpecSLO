"""Independent terminal192 evidence audit, stdlib only.

No actual/receipt input is read until an explicit terminal receipt SHA is supplied.
This is an evidence audit, not a numerical model oracle or an execution permit.
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
P = B / 'tree-live-small-mixed197-cap64-active32-full-v192'
BASE = B / 'tree-high-normal-cap4-native-families-v148'
NATIVE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
ACTION = 'live197-cap64-active32-full463'
ROOT_REVIEW_SHA = '62220f68fc73e3390c0e52ddb48bdede056d853919ad1eb08c17401ca18f7807'
CLOCK_HELPER_SHA = 'd3266647799fc0cf00c49fe98e195ecb85dc446e51ab66c643a5b1d5c3ca0b74'
BACKENDS = ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'transformers', 'safetensors', 'acl')


def need(ok, label):
    if not ok:
        raise ValueError(label)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == '.gz':
        raw = gzip.decompress(raw)
    return json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def desc(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def merged(*maps):
    out = {}
    for values in maps:
        for path, digest in values.items():
            need(path not in out or out[path] == digest, 'conflicting source pin ' + path)
            out[path] = digest
    return out


class Firewall(importlib.abc.MetaPathFinder):
    attempts = []

    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in BACKENDS:
            self.attempts.append(name)
            raise ImportError('Independent192 HOST forbids ' + name)


def function(path, name, namespace):
    nodes = [n for n in ast.walk(ast.parse(Path(path).read_text()))
             if isinstance(n, ast.FunctionDef) and n.name == name]
    need(len(nodes) == 1, 'one original function ' + name)
    node = copy.deepcopy(nodes[0])
    node.decorator_list = []
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(tree, str(path) + '::independent-HOST', 'exec'), namespace)
    return namespace[name]


def verify_archive(path, pins):
    with tarfile.open(path, 'r:gz') as bundle:
        members = bundle.getmembers()
        need(len(members) == len(pins) and len({m.name for m in members}) == len(members),
             'exact archive inventory')
        for member in members:
            name = '/' + member.name.lstrip('/')
            need(member.isfile() and name in pins and
                 hashlib.sha256(bundle.extractfile(member).read()).hexdigest() == pins[name],
                 'exact regular archive member ' + member.name)
    return dict(**desc(path), members=len(pins), all_member_bytes_rechecked=True)


def source_evidence(receipt):
    freeze_path = P / 'FROZEN_SOURCE192.json'
    f = read(freeze_path)
    root_path = Path(f['root_review']['path'])
    need(sha(root_path) == f['root_review']['sha256'] == ROOT_REVIEW_SHA,
         'frozen corrected ROOT review identity')
    root = read(root_path)
    need(root['status'] == 'passed_source_review' and root['candidate'] == str(P), 'ROOT candidate')
    pins = merged(*(f[k] for k in ('files', 'external_files', 'retained_HOST_evidence',
                                   'execution_guard_inputs', 'peer_reviews')),
                  {str(P / 'CPU_RESULT192.json'): f['CPU_sha256']})
    need(root['reviewed_inputs'] == pins, 'ROOT exact source/CPU/guard/peer review map')
    need(all(sha(path) == digest for path, digest in pins.items()), 'current selected frozen bytes')
    need(sha(D / 'analyze_completed_tree_full463_cost.py') == CLOCK_HELPER_SHA,
         'separately pinned independent clock helper')
    need(receipt['readiness']['freeze_sha256'] == sha(freeze_path) and
         receipt['readiness']['CPU_sha256'] == f['CPU_sha256'], 'actual freeze and HOST identity')
    cpu = read(P / 'CPU_RESULT192.json')
    need(cpu['status'] == 'passed' and cpu['tests_run'] == 4 and cpu['failures'] == cpu['errors'] == 0
         and cpu['Torch_imported'] is cpu['NPU_initialized'] is cpu['model_initialized'] is cpu['actual_device_qualification'] is False
         and cpu['schema'] == 'V192.four-small-actual-source-HOST.v1' and cpu['actual_check_count'] == 4, 'retained own4 HOST result')
    # The ROOT-reviewed revision2 closes the original default archive=False gap.
    parent = root['independent_parent_archive_audit']
    need(sha(parent['path']) == parent['sha256'] and read(parent['path']) == parent['complete_evidence'],
         'embedded corrected parent archive audit exact')
    pa = parent['complete_evidence']
    need(pa['status'] == 'passed' and pa['READY_default_archiveFalse_claim_corrected'] is True,
         'corrected parent archive proof, not rejected revision1')
    supplemental = root['archive_revision_peer']
    need(sha(supplemental['path']) == supplemental['sha256'], 'supplemental archive revision peer')
    archived = {}
    old_archives = []
    for item in [pa['base_archive'], *pa['parent_archives']]:
        need(item['all_member_SHA256_checked'] is True and
             item['regular_members'] == len(item['expected_member_map']) and
             sha(item['path']) == item['sha256'], 'same already-member-audited parent binary')
        archived = merged(archived, item['expected_member_map'])
        old_archives.append(dict(path=item['path'], sha256=item['sha256'],
                                 members=item['regular_members'], current_binary_SHA_checked=True,
                                 member_audit_reused_from_exact_ROOT_revision2=True))
    selected = merged(pins, {str(freeze_path): sha(freeze_path), str(root_path): sha(root_path)})
    projected = {path: digest for path, digest in selected.items() if archived.get(path) != digest}
    ar = receipt['readiness']['archive']
    need(len(projected) == ar['members'] == 225 and Path(ar['path']) == P / 'PREPARED_SOURCE_ARCHIVE192.tar.gz'
         and sha(ar['path']) == ar['sha256'], 'current225-member private archive')
    current = verify_archive(ar['path'], projected)
    for k in ('pre_verifier', 'post_verifier'):
        v = receipt[k]
        actual = read(v['path'])
        need(sha(v['path']) == v['sha256'] and v['exit_code'] == 0 and actual == v['actual_result']
             and actual['status'] == 'pass' and actual['errors'] == [] and
             (actual['checked_frozen_files'], actual['checked_external_files'], actual['checked_packages']) ==
             (10496, 15, 212), 'saved original verifier ' + k)
    for k in ('devices_before', 'devices_after'):
        v = receipt[k]
        need(v['all_idle'] is v['four_910B2'] is True and set(v['processes']) == {'2', '5', '6', '7'}
             and all(z['exit_code'] == 0 and z['idle'] is True and
                     z['output'].strip() == 'No process in device.' for z in v['processes'].values()),
             'saved four idle ' + k)
    need(receipt['sources_unchanged'] is True and receipt['owned_remaining'] == [] and
         receipt['execution_lock']['nonblocking_exclusive'] is True and
         receipt['limits'] == dict(child_execution_seconds=7200, actual_tag_bytes=256 * 1024 ** 2,
                                  aggregate_private_bytes=512 * 1024 ** 2),
         'original source/selflock/limits/owned cleanup')
    return dict(freeze=desc(freeze_path), CPU=desc(P / 'CPU_RESULT192.json'),
                ROOT_review=desc(root_path), selected_SHA_checked=len(pins), archive=current,
                corrected_parent_audit=desc(parent['path']), supplemental_peer=desc(supplemental['path']),
                parent_archives=old_archives, no_original_verifier_or_HOST_suite_rerun=True,
                saved_all_four_idle=True, owned_remaining=[])


class Values:
    """Explicit list carrier for the unchanged integer HOST tree verifier only."""
    def __init__(self, values): self.values = values
    @property
    def ndim(self): return 1
    def numel(self): return len(self.values)
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return copy.deepcopy(self.values)
    def __getitem__(self, key): return Values(self.values[key])


def component_extra(directory, original_audit):
    import scope192
    from component192 import COUNTERS, FAILURES
    from execution_schema192 import valid_execution
    reports = [read(directory / 'large197-component' / f'rank{r}-component.json.gz') for r in range(4)]
    torch = types.SimpleNamespace(Tensor=Values, tensor=lambda v, **kw: Values(v), long='long')
    ns = dict(torch=torch, TreeVerificationOutput=lambda a, b:
              types.SimpleNamespace(token_ids=a, accepted_node_indices=b))
    verify = function(NATIVE / 'native_engine.py', '_verify_tree_outputs_host', ns)
    parents = function(NATIVE / 'tree.py', '_spine_first_topology', {})(2, 4)[0]
    counter_subtractions = zero_subtractions = integer_verdicts = owners = 0
    families = set()
    phases = collections.Counter()
    for rank, report in enumerate(reports):
        need(len(report['traces']) == 12, 'four scopes times3 complete histories')
        for trace in report['traces']:
            scope = next(s for s in scope192.SCOPES if s.name == trace['scope'])
            values = trace['Native']
            need(set(values['before']) == set(values['after']) == set(values['delta']) == set(COUNTERS),
                 'all actual21 fields')
            for k in COUNTERS:
                a, z, delta = (values[v][k] for v in ('before', 'after', 'delta'))
                need(type(a) is type(z) is type(delta) is int and 0 <= a <= z and z - a == delta,
                     'independent raw component subtraction')
                need(k not in FAILURES or delta == 0, 'no component Native fallback/failure')
                counter_subtractions += 1
                if trace['pass_index'] > 0:
                    need(delta == 0, 'complete fresh-history component zero')
                    zero_subtractions += 1
            need(trace['duplicate_publication_guard'] is not None and
                 trace['postpublication_normal_append_is_not_reserved_lease_growth'] is True and
                 trace['original_normal_next_draft'] is not None and
                 trace['published_prompt_next_draft'] is not None,
                 'retained real once-publication and both nextdraft probes')
            for record in trace['records']:
                geo = record['geometry']
                raw_phase = directory / 'large197-component' / (
                    f"rank{rank}-pass{trace['pass_index']}-{scope.name}-cycle{record['cycle']}.json.gz")
                need(read(raw_phase) == record, 'every saved phase equals the complete transaction report')
                phases[(scope.name, geo['prompt_queries'], geo['final'])] += 1
                if rank == 0:
                    draft = record['draft_prompt']
                    need(draft['P'] == geo['prompt_queries'] and
                         draft['all_hidden_and56queryKV_original_matcher'] is True and
                         draft['own_prefix_finite_unchanged'] is True and draft['candidate_KV_copied'] is False,
                         'actual own draft prompt fullhidden/all56KV gate')
                    continue
                target = record['target']
                need(target['all_source_prefix_hashes_finite_unchanged'] is True and
                     record['reference']['all_source_prefix_hashes_finite_unchanged'] is True,
                     'actual immutable source prefixes')
                plans = []
                for selected, row, budget in zip(record['selected'], record['rows'], scope.budgets, strict=True):
                    remap = {old: new for new, old in enumerate(selected)}
                    need(len(row) == len(selected) == budget and selected == sorted(set(selected)) and
                         all(parents[i] == -1 or parents[i] in remap for i in selected),
                         'actual selected ancestor-closed raw8 subset')
                    packed = [-1 if parents[i] == -1 else remap[parents[i]] for i in selected]
                    plans.append(types.SimpleNamespace(candidate_budget=budget, parent_indices=Values(packed)))
                output = verify(record['rows'], plans, Values(target['normal_query_tokens']), 4, 'HOST-values')
                need([[t for t in row if t >= 0] for row in output.token_ids.values] ==
                     record['compiled_commit']['committed_tokens'] and output.accepted_node_indices.values ==
                     record['compiled_commit']['accepted_paths'], 'original Native integer verifier vs actual compiled commit')
                if geo['final']:
                    need(target['final_prompt_tokens'] == [record['publication']['first_token']],
                         'published first token equals actual final head')
                else:
                    need(target['final_prompt_tokens'] == [] and record['publication']['published'] is False,
                         'no intermediate prompt token')
                integer_verdicts += 1
                owners += scope.target_requests
                for phase in target['Native_phases']:
                    need(valid_execution(phase['original_execution']), 'strict four-field NativeGraphExecution')
        inv = report['inventory_delta']
        need(set(inv['before']) == set(inv['after']) == {'entries', 'draft_entries', 'target_entries'},
             'actual three Native inventory families')
        for family in inv['before']:
            before, after = set(inv['before'][family]), set(inv['after'][family])
            expected = set(inv['new_keys']) if rank and family == 'target_entries' else set()
            need(before <= after and after - before == expected, 'actual exact new-key inventory subtraction')
        if rank:
            need(set(map(tuple, inv['actual_route_S_families'])) ==
                 {('causal', 19), ('causal', 35), ('FULL', 19), ('FULL', 35)} and
                 len(inv['new_keys']) == inv['new_hidden_keys'] == 4,
                 'four naturally qualified actual197 hidden families')
            parsed = [ast.literal_eval(k) for k in inv['new_keys']]
            need(all(type(k) is tuple and len(k) == 2 and k[1] == (197,) and
                     k[0].startswith('target-hidden:0|steps:1|') for k in parsed),
                 'actual hidden197 producer keys, not logical-role aliases')
            families.update(map(tuple, inv['actual_route_S_families']))
    need(zero_subtractions == 4 * 4 * 2 * 21, 'all new component2zero subtractions')
    return dict(original_audit=original_audit, raw21_subtractions=counter_subtractions,
                raw21_zero_subtractions=zero_subtractions, original_integer_verifier_calls=integer_verdicts,
                original_integer_owner_verdicts=owners, actual_route_S_families=sorted(families),
                actual_phase_histogram=[dict(scope=n, chunk=q, final=f, records=c)
                                        for (n, q, f), c in sorted(phases.items())],
                hidden_KV_recomputed_on_HOST=False, tensor_markers_are_current_runtime_gates=True)


def reserve_summary(trace):
    """Independent single-reservation replay; do not invent a full active history."""
    current = None
    counts = collections.Counter()
    peak_reserved = publication_active_reserved = mixed_target_reserved = 0
    for e in trace['logical_events']:
        counts[e['kind']] += 1
        if e['kind'] == 'reservation':
            need(current is None, 'one reservation at a time')
            current = e['owner']; peak_reserved = 1
        elif e['kind'] == 'plan':
            need(current is not None and current not in e['target_ids'], 'reserved owner excluded from normal target')
            mixed_target_reserved = max(mixed_target_reserved, len(e['target_ids']) + 1)
            need(len(e['target_ids']) + 1 <= 32, 'formal mixed normal plus reserved fits service32')
        elif e['kind'] == 'publication-entry':
            need(e['owner'] == current and current not in e['active_before'], 'publication still owns reservation')
            publication_active_reserved = max(publication_active_reserved, len(e['active_before']) + 1)
            need(len(e['active_before']) + 1 <= 32, 'formal publication active plus reserved fits32')
        elif e['kind'] in ('publication', 'abort'):
            need(current == e['owner'], 'reservation owner removed exactly at publication/abort')
            current = None
    need(current is None, 'terminal reservation closed')
    return dict(logical_event_counts=dict(counts), max_simultaneous_reservations=peak_reserved,
                max_observed_publication_active_plus_reserved=publication_active_reserved,
                max_mixed_target_subset_plus_reserved=mixed_target_reserved,
                retained_active_peak=trace['active_peak'], service_window=32, configured_capacity=64,
                exact_global_active_plus_reserved_peak_available=False,
                retirement_event_count_available=False,
                limitation='Reservation and publication snapshots are replayable. No complete cross-stream retirement/active-list timeline is retained, so these snapshot maxima are not an exact global active-plus-reserved peak. The diagnostic N32 plus reserved fixture deliberately uses capacity64 and is not formal servicewindow32.')


def full_extra(traces):
    t = traces[1]
    ns = dict(math=math, statistics=statistics, CLASSES=('tight', 'normal', 'loose'))
    function(D / 'analyze_completed_tree_full463_cost.py', 'need', ns)
    metrics = function(D / 'analyze_completed_tree_full463_cost.py', 'clocks', ns)(t)
    first = {}
    first_stream, last_stream = {}, {}
    for e in t['stream_events']:
        i = e['request_index']
        first_stream.setdefault(i, e['elapsed_seconds'])
        last_stream[i] = e['elapsed_seconds']
        if i not in first:
            need('first_token_wall_seconds192' in e, 'actual first callback sample')
            first[i] = e['first_token_wall_seconds192']
        else:
            need('first_token_wall_seconds192' not in e, 'only one first callback sample')
    need(set(first) == set(range(463)), 'all463 TTFT samples')
    per_request = []
    for i, row in enumerate(t['generate_results']):
        ttft = 1000 * (first[i] - row['arrival_ts'])
        need(math.isfinite(ttft) and ttft >= 0 and row['arrival_ts'] == t['absolute_arrivals'][i], 'real original TTFT/arrival')
        per_request.append(dict(request_index=i, request_id=row['request_id'], slo_class=row['slo_class'],
                                slo_tpot_ms=row['slo_tpot_ms'], output_tokens=len(row['completion_token_ids']),
                                Native_paper_tpot_ms=row['paper_tpot_ms'], Native_observed_tpot_ms=row['observed_tpot_ms'],
                                host_stream_paper_tpot_ms=1000 * (last_stream[i] - first_stream[i]) / 256,
                                host_stream_observed_tpot_ms=1000 * (last_stream[i] - first_stream[i]) / 255,
                                TTFT_ms=ttft, attained=row['paper_tpot_ms'] <= row['slo_tpot_ms']))
    def percentile(values):
        values = sorted(values)
        at = (len(values) - 1) * .9
        lo, hi = math.floor(at), math.ceil(at)
        return dict(median_ms=statistics.median(values),
                    P90_ms=values[lo] + (values[hi] - values[lo]) * (at - lo))
    metrics['TTFT192'] = dict(samples=463, **percentile([r['TTFT_ms'] for r in per_request]),
                             by_class={cl: percentile([r['TTFT_ms'] for r in per_request if r['slo_class'] == cl])
                                       for cl in ('tight', 'normal', 'loose')})
    plans = [e for e in t['logical_events'] if e['kind'] == 'plan']
    normal = t['normal_target_calls']
    prompt = sum(e['end'] - e['start'] for e in plans)
    mixed_normal = sum(sum(b + 1 for b in e['budgets']) for e in plans)
    mixed_physical = sum(sum(e['query_partition']) for e in plans)
    need(all(sum(e['query_partition']) == 197 and 1 <= e['end'] - e['start'] <= 64 for e in plans), 'actual mixed197/cap64')
    geometry = dict(normal_only_calls=len(normal), mixed197_calls=len(plans),
                    normal_only_owner_rounds=sum(c['N'] for c in normal),
                    mixed_normal_owner_rounds=sum(len(e['target_ids']) for e in plans),
                    normal_only_real_queries=sum(c['real_query_count'] for c in normal),
                    normal_only_physical_queries=sum(c['physical_M'] for c in normal),
                    mixed_normal_queries=mixed_normal, mixed_prompt_queries=prompt,
                    mixed_dummy_queries=mixed_physical - prompt - mixed_normal,
                    mixed_physical_queries=mixed_physical,
                    chunk_size_histogram=dict(sorted(collections.Counter(e['end'] - e['start'] for e in plans).items())),
                    mixed_N_histogram=dict(sorted(collections.Counter(len(e['target_ids']) for e in plans).items())),
                    normal_only_N_histogram=dict(sorted(collections.Counter(c['N'] for c in normal).items())),
                    standalone_prefill_calls=len(t['standalone_prefill_calls']),
                    standalone_prompt_queries=sum(c['uncached_queries'] for c in t['standalone_prefill_calls']))
    zeros = 0
    for v in traces:
        need(v['sealed_before'] is v['sealed_after'] is v['cache_released'] is True, 'all4 full sealed/released')
        before, after, delta = (v['native'][k] for k in ('before', 'after', 'delta'))
        need(len(delta) == 21 and set(before) == set(after) == set(delta), 'actual full21 schema')
        need(all(type(before[k]) is type(after[k]) is type(delta[k]) is int and after[k] - before[k] == delta[k] == 0 for k in delta), 'actual all4 full21zero')
        zeros += len(delta)
    return dict(metrics=metrics, per_request=per_request, geometry=geometry,
                reservation=reserve_summary(t), full21_zero_subtractions=zeros,
                allocator_by_rank={str(v['rank']): v['allocator'] for v in traces},
                HOST_phase_seconds_by_rank={str(v['rank']): v['phase_seconds'] for v in traces},
                decode_counters_leader=t['decode_counters'],
                phases_are_HOST_boundaries_not_additive_device_times=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--terminal-authorized', action='store_true', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--expected-receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.tag) is not None, 'safe terminal tag')
    need(args.output.parent.resolve() == D and not args.output.exists(), 'fresh D-only output')
    receipt_path = P / ('receipt-' + args.tag + '.json')
    need(sha(receipt_path) == args.expected_receipt_sha256, 'explicit terminal receipt SHA')
    receipt = read(receipt_path)
    need(receipt['status'] in ('complete', 'failed'), 'running receipt is forbidden')
    directory = P / 'local-output' / args.tag
    need(directory.is_dir() and directory.resolve() == directory and not directory.is_symlink(), 'own terminal local-output')
    sys.meta_path.insert(0, Firewall())
    errors = []
    result = dict(schema='V192.independent-terminal-HOST-audit.v1', candidate=str(P), tag=args.tag,
                  job_status=receipt['status'], receipt=desc(receipt_path), script=desc(__file__),
                  errors=errors, adopted=False, formal_Goodput=False, Torch_imported=False, NPU_used=False)
    log_path = P / ('local-' + args.tag + '.log')
    need(Path(receipt['child']['log']) == log_path and not log_path.is_symlink(), 'own exact child log')
    result['child_log'] = desc(log_path)
    try:
        result['provenance'] = source_evidence(receipt)
        need(read(directory / 'worker-exit.json') == dict(closed=True, all_exited=True), 'all workers closed')
    except Exception as error:
        errors.append(dict(stage='source_or_terminal_provenance', error=repr(error)))
    paths = sorted(directory.rglob('*'))
    need(not any(f.is_symlink() for f in paths), 'no unexpected terminal raw alias')
    manifest = {str(f): desc(f) for f in paths if f.is_file()}
    manifest_path = args.output.with_name(args.output.stem + '_RAW_MANIFEST.json')
    need(not manifest_path.exists(), 'fresh raw manifest')
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    result['raw_manifest'] = dict(**desc(manifest_path), files=len(manifest), bytes_total=sum(v['bytes'] for v in manifest.values()))
    result['worker_errors'] = [read(f) for f in sorted(directory.glob('rank*-last-error.json'))]
    # No audit module is imported unless the exact frozen source/terminal gates passed.
    if 'provenance' in result:
        sys.path.insert(0, str(P))
        try:
            import audit192
            reports = [read(directory / f'rank{r}-worker.json') for r in range(4)]
            actual = audit192.audit_worker_reports(reports, directory)
            completed = read(directory / 'completed.json')
            need(actual == completed['audit'] == receipt['actual_full_audit'] and actual['status'] == 'passed',
                 'unchanged original192 audit equals runtime and supervisor')
            need(completed['action'] == ACTION and completed['replies'] ==
                 [['calibrated', r, dict(full463=True, traces=4)] for r in range(4)], 'all4 actual complete replies')
            result['original_unmodified_audit'] = dict(source=desc(P / 'audit192.py'), result=actual,
                                                       equals_completed_and_supervisor=True)
            result['component_extra'] = component_extra(directory, actual['large_mixed_component'])
            zero_count = 0
            for report in reports:
                for item in report['traces']:
                    if item['role'] == 'warmup65':
                        continue
                    raw = read(item['artifact']['path'])
                    before, after, delta = (raw['native'][k] for k in ('before', 'after', 'delta'))
                    need(len(delta) == 21 and set(before) == set(after) == set(delta) and
                         all(type(before[k]) is type(after[k]) is type(delta[k]) is int and
                             after[k] - before[k] == delta[k] == 0 for k in delta),
                         'independent every-rank twozero/full Native subtraction')
                    zero_count += len(delta)
            need(zero_count == 4 * 3 * 21, 'all252 actual twozero/full subtractions')
            result['independent_twozero_and_full_Native_subtractions'] = zero_count
            need(receipt['status'] == 'complete' and receipt['child']['exit_code'] == 0 and
                 receipt['errors'] == [] and receipt['actual_execution_complete'] is True,
                 'successful job remains separate from retained inference')
        except Exception as error:
            errors.append(dict(stage='original_component_full_audit_or_job_completion', error=repr(error)))
        try:
            traces = [read(directory / 'candidate' / f'rank{r}-full463-actual-trace.json.gz') for r in range(4)]
            result['retained_full463'] = full_extra(traces)
        except Exception as error:
            errors.append(dict(stage='retained_full463_independent_arithmetic', error=repr(error)))
    need(not Firewall.attempts and not any(n.split('.')[0] in BACKENDS for n in sys.modules), 'no backend import')
    result['status'] = ('passed_terminal_component_and_full463_audit' if not errors else 'terminal_evidence_with_audit_errors')
    result['limitations'] = [
        'HOST reexecutes source/arrival/cursor/integer verifier/clock arithmetic. It does not reconstruct model hidden/KV tensors from gate booleans; current original tensor checks remain a separately bound runtime qualification.',
        'NativeGraphExecution is exactly mode/fallback_reason/capture_attempted/replay_executed. No missing used_aclgraph field or Boolean default is invented.',
        'Diagnostic N32 normal owners plus one reservation uses capacity64. Formal service is active-plus-reserved32; queued463/storage527/scratch64 have different meanings.',
        'Original192 schedule is one warm65, two complete zero65, one seal, full463. There is no max8 fixedpoint loop; ROOT explicitly accepted fail-closed twozero behavior.',
        'Old BASE/P163/P172 binaries are SHA-rechecked against the embedded corrected ROOT revision2 exact member audit, not re-extracted into a new epoch. New225 archive members are independently checked.',
        'A retained full inference or valid numerical component does not change a failed job into a formal winner. No full-workload independent AR/adoption or guaranteed gain is claimed.',
        'Physical query volumes and summed HOST boundary seconds are descriptive. They do not identify additive device costs or establish causality across runs.'
    ]
    result['UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(**desc(args.output), status=result['status'], job_status=result['job_status'], errors=errors)))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
