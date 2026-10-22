"""Read-only HOST audit of one actual V96 attempt, after its terminal receipt.

No runtime installer, model, torch, device query, or worker is imported.  The
frozen HOST report validators are consumed as source-bound validators, not as
substitutes for the separate numerical, provenance and cleanup evidence.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import importlib.abc
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import sys
import tarfile
import traceback

BASE = Path('/root/SpecSLO/budget-calibration-20261015')
SOURCE = BASE/'tree-capacity64-runtime-v96'
MANIFEST = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/manifest.jsonl')
PRE_AUDIT = BASE/'budget-diagnostics/ROOT_V96_PRE_NPU_SOURCE_AND_HOST_AUDIT.json'
FREEZE_SHA = '6925efa73af1874ebbdac5629bc3e8fb73ce838e73eff3d7121aeb19c411818d'
CPU_SHA = 'ee2433561f9c2227412ede8666c9e1d1834c004f730adfe460b5b0b5f5678042'
ARCHIVE_SHA = 'b9bb1544726669c05c7937b5a3fc182e57ac6349513aa30a0cf54f6677de83d7'
MAX_COMPRESSED = 64*1024**2
MAX_UNCOMPRESSED = 256*1024**2
ROLES = ('warmup65', 'zero65-1', 'zero65-2', 'full463')
SLOS = {'tight': 40., 'normal': 50., 'loose': 150.}
FUSION_PASSES = {'TransdataCastFusionPass', 'TransdataFz2FzgFusionPass', 'TransdataFzg2FzFusionPass'}


class Rejected(ValueError):
    pass


def require(value, message):
    if not value:
        raise Rejected(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def finite_json(value):
    if type(value) is dict:
        for child in value.values():
            finite_json(child)
    elif type(value) is list:
        for child in value:
            finite_json(child)
    elif type(value) is float:
        require(math.isfinite(value), 'Nonfinite actual JSON value')
    else:
        require(value is None or type(value) in (bool, int, str), 'Unexpected JSON value')


def read_json(path):
    value = json.loads(Path(path).read_bytes())
    finite_json(value)
    return value


def descriptor(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha(path))


def read_gzip(artifact, directory):
    root, path = Path(directory).resolve(), Path(artifact['path']).resolve()
    require(path.is_relative_to(root) and path.is_file(), 'Artifact escaped actual run directory')
    raw = path.read_bytes()
    require(0 < len(raw) <= MAX_COMPRESSED and len(raw) == artifact['bytes']
            and sha(raw) == artifact['sha256'], 'Artifact compressed bytes/SHA mismatch')
    with gzip.open(path, 'rb') as stream:
        unpacked = stream.read(MAX_UNCOMPRESSED+1)
    require(len(unpacked) <= MAX_UNCOMPRESSED and len(unpacked) == artifact['uncompressed_bytes'],
            'Artifact uncompressed length mismatch or bound exceeded')
    value = json.loads(unpacked)
    finite_json(value)
    return value


def recover_gzip(path, directory):
    """A raw failed-attempt artifact remains original evidence, not a report."""
    item = descriptor(path)
    require(0 < item['bytes'] <= MAX_COMPRESSED, 'Raw recovery gzip exceeds bound')
    with gzip.open(path, 'rb') as stream:
        raw = stream.read(MAX_UNCOMPRESSED+1)
    require(len(raw) <= MAX_UNCOMPRESSED, 'Raw recovery expansion exceeds bound')
    item['uncompressed_bytes'] = len(raw)
    return item, read_gzip(item, directory)


class HostImportsOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend')
                or fullname.endswith(('.native_engine', '.native_model', '.native_graph'))):
            raise Rejected('HOST auditor forbids model/device import: '+fullname)
        return None


@contextmanager
def frozen_host_modules(source=SOURCE):
    guard, previous = HostImportsOnly(), list(sys.path)
    sys.meta_path.insert(0, guard)
    sys.path.insert(0, str(source))
    try:
        names = ('audit_capacity64_v96', 'bounded_report_audit_v78', 'finite_inventory_plan',
                 'startup_receipts_v75', 'local_qualification_scope')
        modules = {}
        for name in names:
            module = importlib.import_module(name)
            require(Path(module.__file__).resolve().parent == Path(source).resolve(),
                    'HOST validator imported outside frozen source: '+name)
            modules[name] = module
        yield modules
    finally:
        sys.path[:] = previous
        sys.meta_path.remove(guard)


def percentile_linear(values, probability):
    """Linear interpolation, index=(n-1)*p (NumPy default linear method)."""
    values = sorted(values)
    require(values and 0 <= probability <= 1 and all(type(v) in (int,float)
            and math.isfinite(v) for v in values), 'Invalid percentile population')
    point = (len(values)-1)*probability
    lo, hi = math.floor(point), math.ceil(point)
    return values[lo]+(values[hi]-values[lo])*(point-lo)


def production_tpot_contract(source=SOURCE):
    path=Path(source)/'runtime_tree_decode.py'
    tree=ast.parse(path.read_text())
    expected={
        'request_decode_elapsed_ms':'finished_decode_elapsed.get(index, states[index].decode_elapsed_ms)',
        'measured_decode_elapsed_ms':'max(0.0, request_decode_elapsed_ms - (states[index].decode_start_elapsed_ms or 0.0))',
        'observed_tpot_ms':'measured_decode_elapsed_ms / max(1, len(completion) - 1)',
        'paper_tpot_ms':'measured_decode_elapsed_ms / max(1, len(completion))'}
    assignments={}
    for name,expression in expected.items():
        matches=[node for node in ast.walk(tree) if isinstance(node,ast.Assign)
                 and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id==name]
        require(len(matches)==1 and ast.dump(matches[0].value)==ast.dump(ast.parse(expression,mode='eval').body),
                'Pinned production TPOT AST formula differs: '+name)
        assignments[name]=dict(line=matches[0].lineno,expression=ast.unparse(matches[0].value))
    outputs=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
             and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
             and node.func.value.id=='results' and node.func.attr=='append'
             and len(node.args)==1 and isinstance(node.args[0],ast.Dict)]
    require(len(outputs)==1,'Unique original result materialization not found')
    keys=[key.value for key in outputs[0].args[0].keys if isinstance(key,ast.Constant)]
    require('paper_tpot_ms' in keys and 'observed_tpot_ms' in keys
            and 'decode_start_elapsed_ms' not in keys and 'finished_decode_elapsed_ms' not in keys,
            'Current original result clock availability differs')
    return dict(passed=True,source=descriptor(path),assignments=assignments,
        raw_finish_start_clocks_available_in_results=False,
        independent_raw_clock_reconstruction_qualified=False,
        boundary='Source AST proves the production formula; actual reported TPOT pair can be checked for denominator consistency, but omitted raw finish/start clocks cannot be reconstructed independently')


def metric_rows(trace, manifest_rows):
    """Recompute attainment/GP from actual TPOT records and original outputs.

    V96 does not retain raw finish/start clocks. The paper/observed pair is
    checked against its source-bound denominator relation, never relabeled as
    an independent raw-clock measurement.
    """
    results, wall = trace['generate_results'], trace['host_generate_call_seconds']
    require(type(results) is list and len(results) == len(manifest_rows)
            and type(wall) in (int,float) and math.isfinite(wall) and wall > 0
            and trace['elapsed_seconds'] == wall, 'Real host generate timer/results required')
    delivered=[[] for _ in manifest_rows]
    delivery_times=[[] for _ in manifest_rows]
    finished=[0]*len(manifest_rows)
    previous=-1.
    for event in trace['stream_events']:
        owner=event['request_index'];elapsed=event['elapsed_seconds'];tokens=event['token_ids']
        require(type(owner) is int and 0<=owner<len(manifest_rows)
                and event['request_id']==manifest_rows[owner]['request_id'] and not finished[owner]
                and type(tokens) is list and tokens and all(type(t) is int and t>=0 for t in tokens)
                and type(elapsed) in (int,float) and math.isfinite(elapsed)
                and elapsed>=previous and type(event['finished']) is bool,
                'Actual stream owner/token/time/terminal record differs')
        previous=elapsed
        delivered[owner].extend(tokens);delivery_times[owner].append(elapsed)
        finished[owner]+=int(event['finished'])
    require(wall+1.e-6>=previous,'Real host timer excludes last original stream output')
    records, classes = [], {}
    for index, (result, request) in enumerate(zip(results, manifest_rows, strict=True)):
        tokens = result['completion_token_ids']
        require(len(tokens) == 256 and all(type(t) is int and t >= 0 for t in tokens)
                and result['request_id'] == request['request_id'] and delivered[index]==tokens
                and finished[index]==1 and delivery_times[index], 'Original stream/result complete256 mismatch')
        paper,observed=result['paper_tpot_ms'],result['observed_tpot_ms']
        require(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in (paper,observed))
                and math.isclose(paper,observed*255/256,rel_tol=1.e-10,abs_tol=1.e-8),
                'Actual paper/observed TPOT denominator consistency differs')
        cls = request['slo_class']
        require(cls in SLOS and request['slo_tpot_ms'] == SLOS[cls], 'Manifest SLO differs')
        attained = paper <= SLOS[cls]
        require(result['paper_slo_attained'] is attained
                and result['paper_slo_goodput_tokens'] == (len(tokens) if attained else 0),
                'Paper attainment differs from actual TPOT/SLO arithmetic')
        stream_ms=(delivery_times[index][-1]-delivery_times[index][0])*1000
        row = dict(request_index=index, request_id=request['request_id'], slo_class=cls,
            output_tokens=len(tokens),
            paper_tpot_ms=paper, observed_tpot_ms=observed, paper_slo_attained=attained,
            paper_good_tokens=len(tokens) if attained else 0,
            stream_first_to_last_delivery_ms=stream_ms,
            stream_delivery_wall_TPOT_ms=stream_ms/255)
        records.append(row)
        classes.setdefault(cls, []).append(row)
    def aggregate(rows):
        output, good = sum(r['output_tokens'] for r in rows), sum(r['paper_good_tokens'] for r in rows)
        paper, observed = ([r[key] for r in rows] for key in ('paper_tpot_ms', 'observed_tpot_ms'))
        return dict(requests=len(rows), attained=sum(r['paper_slo_attained'] for r in rows),
            output_tokens=output, paper_good_tokens=good, paper_goodput_tokens_per_second=good/wall,
            raw_tokens_per_second=output/wall, paper_TPOT_ms_median=statistics.median(paper),
            paper_TPOT_ms_P90=percentile_linear(paper,.9),
            observed_TPOT_ms_median=statistics.median(observed),
            observed_TPOT_ms_P90=percentile_linear(observed,.9),
            stream_delivery_wall_TPOT_ms_median=statistics.median(r['stream_delivery_wall_TPOT_ms'] for r in rows),
            stream_delivery_wall_TPOT_ms_P90=percentile_linear([r['stream_delivery_wall_TPOT_ms'] for r in rows],.9))
    return dict(**aggregate(records), elapsed_seconds=wall,
        native_inner_elapsed_seconds=trace.get('native_elapsed_seconds'),
        timing_scope='Worker original Native generate_batch call entry to return; original arrivals included; startup/tokenization/three65 traces excluded',
        by_class={name: aggregate(rows) for name,rows in classes.items()}, request_metrics=records,
        methods=[dict(quantity='paper_TPOT_ms',method='Actual Native reported paper_tpot_ms; pinned source formula uses charged decode-after-first-token ms / 256; raw finish/start clocks absent'),
                 dict(quantity='observed_TPOT_ms',method='Actual Native reported observed_tpot_ms; independently check paper == observed * 255/256'),
                 dict(quantity='paper_Goodput',method='sum(256 for each paper_TPOT_ms <= original class SLO) / actual host generate call seconds'),
                 dict(quantity='raw_tokens_per_second',method='actual complete output tokens / same actual host generate call seconds'),
                 dict(quantity='stream_delivery_wall_TPOT_ms',method='(last original stream delivery seconds - first delivery seconds) * 1000 / 255; auxiliary wall delivery clock, different from Native charged paper clock'),
                 dict(quantity='median',method='ordinary sample median; midpoint average for an even population'),
                 dict(quantity='P90',method='sorted ascending; linearly interpolate at zero-based (n-1)*0.9')],
        formal_Goodput=False, outer_driver_E2E_baseline_ratio_qualified=False,
        raw_finish_start_clocks_available=False,independent_raw_clock_reconstruction_qualified=False,
        whole_workload_AR_equality_qualified=False)


def fusion_descriptor(path):
    require(0 < Path(path).stat().st_size <= MAX_COMPRESSED, 'Fusion diagnostic size invalid')
    pairs = json.loads(Path(path).read_text(), object_pairs_hook=lambda pairs:pairs)
    require(type(pairs) is list and pairs, 'Fusion top-level malformed')
    for name, record in pairs:
        require(re.fullmatch(r'session_and_graph_id_\d+_\d+', name)
                and type(record) is list and len(record) == 1 and record[0][0] == 'graph_fusion',
                'Fusion session contains unexpected data')
        values = record[0][1]
        require(type(values) is list and values and len({k for k,_ in values}) == len(values),
                'Fusion pass list malformed')
        for kind, counters in values:
            require(kind in FUSION_PASSES and type(counters) is list and len(counters) == 2
                    and {k for k,_ in counters} == {'effect_times','match_times'}
                    and all((type(v) is int and v >= 0) or (type(v) is str and re.fullmatch(r'\d+',v))
                            for _,v in counters), 'Fusion pass/counter contains unexpected data')
    return dict(**descriptor(path), kind='actual-CANN-graph-fusion-counts', graph_records=len(pairs))


def check_source_inventory(source, frozen):
    names = read_json(source/'STATIC_INPUTS_V96.json')['files']
    require(names == sorted(set(names)) and len(names) == 208
            and 'STATIC_INPUTS_V96.json' in names, 'Explicit static208 inventory differs')
    require({str((source/name).resolve()) for name in names} == set(frozen), 'Freeze/source inventory mismatch')
    fusion = []
    evidence = {'CPU_RESULT_V96.json','PREPARED_SOURCE_HASHES_V96.json'}
    # Keep this recognition independent from the candidate collector. Nothing
    # is deleted, ignored by a broad glob, or silently made new source.
    allowed = set(names) | evidence
    for path in source.iterdir():
        if not path.is_file() or path.suffix not in ('.py','.sh','.json') or path.name in allowed:
            continue
        if path.name == 'fusion_result.json':
            fusion.append(fusion_descriptor(path))
        elif path.name.startswith('CPU_ATTEMPT_') and path.suffix == '.json':
            require(read_json(path).get('schema') == 'V96.actual-HOST-source-and-contract-tests.v1',
                    'Unknown HOST-attempt output structure')
        elif re.fullmatch(r'local-receipt-[A-Za-z0-9][A-Za-z0-9._-]*\.json',path.name):
            value = read_json(path)
            require(value.get('schema') == 'V96.capacity64-runtime-supervisor.v1'
                    and value.get('action') in ('capacity64-qualify65','capacity64-full463'),
                    'Unknown supervisor output structure')
        else:
            raise Rejected('Unknown unbound static input: '+str(path))
    return fusion


def audit_archive(path, expected):
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        wanted = {Path(name).name: digest for name,digest in expected.items()}
        require(len(wanted) == len(expected) and len(members) == len(wanted)
                and len({m.name for m in members}) == len(wanted)
                and {m.name for m in members} == set(wanted)
                and all(m.isfile() for m in members), 'Archive exact regular-member inventory differs')
        for member in members:
            require(sha(archive.extractfile(member).read()) == wanted[member.name],
                    'Archive member bytes differ: '+member.name)
    return dict(**descriptor(path), unique_regular_members=len(wanted), all_member_bytes_match=True)


def audit_sources(source, receipt):
    freeze_path, cpu_path = source/'PREPARED_SOURCE_HASHES_V96.json', source/'CPU_RESULT_V96.json'
    require(file_sha(freeze_path) == FREEZE_SHA and file_sha(cpu_path) == CPU_SHA,
            'Pinned V96 freeze/CPU identity differs')
    freeze, cpu = read_json(freeze_path), read_json(cpu_path)
    require(freeze['CPU_sha256'] == CPU_SHA and len(freeze['files']) == 208
            and len(freeze['external_files']) == 1812 and freeze['actual_HOST_tests'] == 172,
            'Pinned source/CPU closure counts differ')
    require(cpu['status'] == 'passed' and cpu['tests_run'] == 172 and cpu['errors'] == cpu['failures'] == 0
            and cpu['initialization_attempts'] == [] and cpu['NPU_initialized'] is False
            and cpu['input_hashes'] == cpu['input_hashes_after'] == freeze['files']
            and cpu['external_input_hashes'] == cpu['external_input_hashes_after'] == freeze['external_files'],
            'Actual CPU result/source before-after closure rejected')
    for name, digest in {**freeze['files'], **freeze['external_files']}.items():
        require(file_sha(name) == digest, 'Actual source/external SHA mismatch: '+name)
    fusion = check_source_inventory(source, freeze['files'])
    expected = dict(freeze['files']) | {str(freeze_path):FREEZE_SHA, str(cpu_path):CPU_SHA}
    archive_path = source/'PREPARED_SOURCE_ARCHIVE_V96.tar.gz'
    require(file_sha(archive_path) == ARCHIVE_SHA, 'Pinned prepared archive SHA differs')
    prepared = audit_archive(archive_path,expected)
    snapshots = []
    if 'source_hashes' in receipt:
        require(receipt['source_hashes'] == expected, 'Actual supervisor source snapshot map differs')
        snapshots.append(audit_archive(Path(receipt['source_snapshot']),expected))
        require(snapshots[-1]['sha256'] == receipt['source_snapshot_sha256'], 'Actual snapshot SHA mismatch')
    if 'external_input_hashes' in receipt:
        require(all(receipt['external_input_hashes'].get(name) == digest
                    for name,digest in freeze['external_files'].items()), 'Actual supervisor external closure differs')
        for name,digest in receipt['external_input_hashes'].items():
            require(file_sha(name) == digest, 'Supervisor actual external after-SHA differs: '+name)
    pre = read_json(PRE_AUDIT)
    require(pre['passed'] is True and pre['freeze_sha256'] == FREEZE_SHA
            and pre['archive_sha256'] == ARCHIVE_SHA, 'Independent real prelaunch audit identity differs')
    runtime = BASE/'serial-v1/runtime.sh'
    sourced = re.findall(r'^source\s+([^\s]+)', runtime.read_text(), flags=re.M)
    shells_before = pre['actual_sourced_runtime_shells_before']
    require(len(sourced) == 2 and set(sourced) == set(shells_before), 'Actual runtime source paths differ from pre-record')
    shells = []
    for path in sourced:
        after = file_sha(path)
        require(after == shells_before[path], 'Actually sourced runtime shell after-SHA differs: '+path)
        shells.append(dict(path=path, resolved_after=str(Path(path).resolve()),
            before_sha256=shells_before[path], after_sha256=after, unchanged=True))
    return dict(passed=True, local_files=208, external_files=1812, CPU=descriptor(cpu_path),
        freeze=descriptor(freeze_path), prepared_archive=prepared, actual_snapshots=snapshots,
        actual_prelaunch_audit=descriptor(PRE_AUDIT), actual_sourced_shells=shells,
        actual_runtime_outputs=fusion, unknown_static_inputs_rejected=True,
        source_and_CPU_provenance_is_device_qualification=False)


def audit_startup(reports, modules, source):
    plan, startup_module = modules['finite_inventory_plan'], modules['startup_receipts_v75']
    explicit = {kind: set(map(repr, keys)) for kind,keys in plan.draft_keys().items()}
    generic = set(map(repr,startup_module.original_draft_family_keys()))
    target = {f['expected_key'] for f in plan.targets()}
    queries = [r['query_count'] for r in read_json(source/'prompt-shape-examples.json')['examples']]
    prefill = set(map(repr,startup_module.expected_prefill_keys(queries)))
    summaries = []
    for rank, report in sorted(reports.items()):
        row = report['startup']
        require(row['rank'] == rank and report['startup_completed'] is True, 'Actual startup rank/completion differs')
        keys = row['actual_native_keys']
        require(set(keys) == {'entries','draft_entries','target_entries'} and keys['entries'] == []
                and all(len(v) == len(set(v)) for v in keys.values()), 'Actual mapping keys aliased/missing')
        expected = dict(entries=set(), draft_entries=explicit['chain'],
            target_entries=explicit['hidden_PA'] | explicit['frontier_PA'] | generic) if rank == 0 else dict(
                entries=set(), draft_entries=set(),target_entries=target | prefill)
        require({k:set(v) for k,v in keys.items()} == expected
                and row['actual_native_entry_counts'] == {k:len(v) for k,v in expected.items()},
                'Actual Native complete key inventory differs')
        mc2 = row['actual_MC2_preflight']
        require(mc2['rank'] == rank and mc2['role'] == ('draft' if rank == 0 else 'target')
                and mc2['all4rank_contract_passed'] is True and mc2['failed_checks'] == []
                and mc2['native_qualification_flags_modified'] is False
                and sorted(mc2['planned_full_chain_rows']) == sorted(plan.MC2_ROWS), 'Actual MC2 preflight incomplete')
        if rank:
            require(mc2['manifest_route_count'] == 128 and mc2['disabled_routes'] == []
                    and mc2['routes_with_missing_or_unqualified_rows'] == [], 'Target MC2 route contract incomplete')
        details = row['original_startup_receipts']
        if rank == 0:
            draft = details['draft']
            require(draft['status'] == 'complete' and draft['explicit_keys'] == 86
                    and {k:set(v) for k,v in draft['native_keys'].items()} == explicit
                    and draft['original_native_flags_unchanged'] is True, 'Actual explicit86 startup incomplete')
        else:
            actual_prefill, actual_target = details['prefill'], details['target']
            require(actual_prefill['status'] == 'complete' and actual_prefill['completed'] == actual_prefill['total'] == 201
                    and actual_prefill['transition_checks_passed'] == 8
                    and actual_target['status'] == 'complete' and actual_target['fixtures'] == 44
                    and set(actual_target['native_keys']) == target
                    and actual_target['original_native_flags_unchanged'] is True, 'Actual original201/44 startup incomplete')
        family = row['original_draft_family']
        delta = {k: family['native_after'][k]-v for k,v in family['native_before'].items()}
        require(delta == family['native_delta'] and all(type(v) is int and v >= 0 for v in delta.values())
                and family['native_qualification_flags_modified'] is False
                and family['original_native_method_executed'] is (rank == 0)
                and family['actual_added_qualified_entries'] == (28 if rank == 0 else 0)
                and family['actual_native_entry_counts'] == row['actual_native_entry_counts'],
                'Original generic28 actual changed/qualification evidence incomplete')
        require(set(family['added_native_keys']['target_entries']) == (generic if rank == 0 else set())
                and family['added_native_keys']['entries'] == family['added_native_keys']['draft_entries'] == [],
                'Actual generic28 additions differ')
        for key in ('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays'):
            require(delta[key] == (28 if rank == 0 else 0), 'Generic28 original qualification counter differs')
        require(all(row['actual_native_counters'][name] == 0 for name in startup_module.FAILURE_COUNTERS),
                'Startup Native failure/fallback counter is nonzero')
        modules['bounded_report_audit_v78'].memory_record(row['actual_original_memory_finite_vote'],rank,'V80_original_startup')
        summaries.append(dict(rank=rank, entries={k:len(v) for k,v in expected.items()},
            explicit_draft=86 if rank == 0 else 0, generic_draft=28 if rank == 0 else 0,
            exact_prefill=201 if rank else 0, target_decode=44 if rank else 0,
            original_native_runtime_validated_startup_mapping_observed=True,
            source_bound_actual_matcher_gates=True, raw_tensor_recomparison_available=False))
    return dict(passed=len(reports)==4, available_ranks=sorted(reports), ranks=summaries,
        new_capacity64_graph_execution_not_inferred_from_old32_receipts=True)


def phase_summary(phase, rank):
    geometry = phase['geometry']
    numeric = phase['target_numeric']
    if rank == 0:
        for field in ('all_physical_first_hidden_original_matcher','all_physical_tail_hidden_original_matcher',
                      'exact_physical_top2_confidence','exact_public_rows_roots_node_confidences',
                      'all28_query_KV_original_matcher','independent_per_owner_ledger'):
            require(phase['draft_numeric'].get(field) is True,'Draft numerical proof missing: '+field)
        require(phase['draft_prompt_numeric']['all28_query_KV_original_matcher'] is True,
                'Draft prompt28 KV proof missing')
    else:
        for field in ('passed','all197_hidden_original_matcher','all64_target_query_KV_original_matcher',
                      'exact_tree_and_final_prompt_heads','all_candidate_and_reference_prefixes_finite_and_unchanged',
                      'actual_query_KV_restored','reference_never_initialized_from_candidate_KV'):
            require(numeric.get(field) is True,'Target numerical proof missing: '+field)
        require(len(numeric['independent_eager_calls']) == 19,'Missing independent FULL retained spans')
    return dict(phase_index=phase['phase_index'], N=geometry['N'], M=geometry['physical_M'],
        S=geometry['physical_S'], mask_shape=geometry['mask_shape'], prompt_start=geometry['prompt_start'],
        prompt_end=geometry['prompt_end'], final=geometry['final'],
        committed_cursor=phase['actual_committed_cursor'], publication=phase['publication']['published'],
        all_layer_KV_matcher_observed=28 if rank == 0 else 64,
        original_changed_restored_gate_observed=rank != 0)


def audit_same_m(chains, reports, modules):
    validator = modules['bounded_report_audit_v78']
    signatures, summaries, identity_epochs = {}, [], []
    for rank, chain in sorted(chains.items()):
        validator.finite_json(chain)
        signatures[rank] = validator.chain_record(chain,'draft' if rank == 0 else 'target',rank=rank)
        phases = []
        for scope in chain['scopes']:
            identity = scope['reservation_identity']
            identity_epochs.append((rank,identity['allocation_epoch']))
            phases.extend(phase_summary(phase,rank) for phase in scope['phases'])
        require(len(phases) == 5, 'Exactly five progressive phases per rank required')
        if rank in reports:
            outer = reports[rank]['mixed_same_M_qualification']
            require(outer['M'] == 197 and outer['S'] == 19 and outer['configured_capacity'] == 64
                    and outer['actual_original_actor_executed'] is True and outer['independent_epochs'] is True
                    and outer['candidate_KV_used_as_reference'] is False, 'Capacity64 actual same-M envelope differs')
            native = outer['native']
            delta = {k:native['after'][k]-v for k,v in native['before'].items()}
            require(delta == native['delta'] and all(v >= 0 for v in delta.values()), 'Same-M Native raw delta differs')
            require(all(delta[k] == 0 for k in modules['startup_receipts_v75'].FAILURE_COUNTERS),
                    'Same-M Native failure/fallback counter nonzero')
        summaries.append(dict(rank=rank, progressive_phases=phases, lifecycle_controls=4,
            target_layer_KV=64 if rank else 0,draft_layer_KV=28 if rank == 0 else 0,
            exact_commit_publication_next_draft=True, independent_source_epochs=True))
    require(not signatures or all(signature == next(iter(signatures.values())) for signature in signatures.values()),
            'Original same-M four-rank tokens/commit/publication disagree')
    return dict(passed=len(chains)==4, available_ranks=sorted(chains), ranks=summaries,
        source_bound_frozen_HOST_chain_validator_used=True,
        qualification='Two genuine-source N15/N16 scopes, five FULL197/S19 progressive phases; independent history plus original matcher/exact heads, KV, changed-restored, physical compaction, commit, publication and next-draft controls',
        offline_tensor_recomparison=False,
        limitation='Artifacts retain actual actor comparator observations and raw counters, not full hidden/KV tensors. Initial-prefix device certificates are checked inside the source-bound actor and are not separately serialized.',
        whole_workload_AR_equality_qualified=False, capacity64_all_geometries_numerically_qualified=False,
        online_qualified=False, formal_Goodput=False)


def audit_progress(source, modules):
    results = []
    for rank in range(4):
        path = source/'startup-progress'/f'finite-inventory-rank{rank}.jsonl'
        if not path.exists():
            results.append(dict(rank=rank,available=False));continue
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        for row in rows:
            finite_json(row)
            modules['bounded_report_audit_v78'].memory_record(row,rank,row['phase'])
        phase_counts = Counter(row['phase'] for row in rows)
        for name,total in (('draft',86),('target_decode',44)):
            records = [row for row in rows if row['phase'] == name]
            if records and records[-1]['completed'] == total:
                require([r['completed'] for r in records] == list(range(total+1))
                        and all(r['total'] == total for r in records), 'Inventory per-shape progress coverage differs')
        drafts=[r for r in rows if r['phase']=='draft' and r['completed']]
        if len(drafts)==86:
            plan=modules['finite_inventory_plan']
            wanted=[('chain',n,k) for n in plan.CHAIN_BUCKETS for k in range(1,5)]
            wanted += [('hidden_PA',n,1) for n in plan.PA_BUCKETS]
            wanted += [('frontier_PA',n,1) for n in plan.FRONTIER_BUCKETS]
            require([(r['detail']['kind'],r['detail']['rows'],r['detail']['steps']) for r in drafts]==wanted,
                    'Actual draft86 progression lacks exact capacity64 shape witnesses')
        target_rows=[r for r in rows if r['phase']=='target_decode' and r['completed']]
        if rank and len(target_rows)==44:
            wanted={r['expected_key']:r for r in modules['finite_inventory_plan'].targets()}
            require({r['detail']['expected_key'] for r in target_rows}==set(wanted),
                    'Actual target44 progress keys differ')
            for record in target_rows:
                detail=record['detail'];fixture=wanted[detail['expected_key']]
                require(all(detail[k]==v for k,v in fixture.items())
                        and detail['semantic_anchors_checked']==(4 if detail['family']=='full9' else 1)
                        and detail['alternate_witness_checked'] is (fixture['alternate_witness'] is not None),
                        'Actual target44 semantic anchor/alternate witness coverage differs')
                partitions=detail['native_partitions_checked']
                witnesses=[('primary',fixture,k) for k in (range(4) if fixture['family']=='full9' else (0,))]
                if fixture['alternate_witness'] is not None:witnesses.append(('alternate',fixture['alternate_witness'],0))
                require(len(partitions)==len(witnesses),'Actual target exact-length variants missing')
                for actual,(label,witness,anchor) in zip(partitions,witnesses,strict=True):
                    ends=[]
                    for q in witness['physical_query_lengths']:ends.append((ends[-1] if ends else 0)+q)
                    require(actual==dict(witness=label,anchor=anchor,actual_seq_lengths_q=ends,
                                         sequence_lens=witness['physical_KV_lengths']),
                            'Actual target Q/KV partition differs')
                dispatch=detail['actual_host_MC2_dispatch_delta']
                require(dispatch['native_epilogue_success']>0 and dispatch['fallback']==dispatch['exception']==0,
                        'Actual target44 original eager matcher had MC2 fallback')
        complete=all([r['completed'] for r in rows if r['phase']==name]==list(range(total+1))
                     for name,total in (('draft',86),('target_decode',44)))
        results.append(dict(rank=rank,available=True,inventory_progress_complete=complete,
            artifact=descriptor(path),records=len(rows),
            phase_counts=dict(phase_counts), memory_maxima={key:max(r['memory'][key] for r in rows)
                for key in ('npu_reserved_bytes','npu_peak_allocated_bytes','graph_entries')},
            minimum_npu_free_bytes=min(r['memory']['npu_free_bytes'] for r in rows)))
    return dict(passed=all(row.get('inventory_progress_complete') is True for row in results),
        ranks=results, all_available=all(row['available'] for row in results))


def audit_cleanup(receipt, directory):
    require(receipt['status'] in ('complete','failed'), 'Actual attempt is not terminal')
    errors = []
    def actual_hardware_idle(record):
        if not isinstance(record,dict):return False
        info=record.get('hardware',{})
        parsed=[(int(card),name) for card,name in re.findall(r'^\|\s*(\d+)\s+(\S+)\s*\|',
                    info.get('output',''),re.M) if not name.isdecimal()]
        processes=record.get('processes',{})
        return bool(info.get('exit_code')==0 and not info.get('timed_out')
            and sorted(parsed)==[(card,'910B2') for card in (2,5,6,7)]
            and set(processes)=={'2','5','6','7'} and record.get('all_idle') is True
            and record.get('four_910B2') is True and sorted(map(tuple,record.get('parsed_hardware',[])))==sorted(parsed)
            and all(row.get('exit_code')==0 and not row.get('timed_out') and row.get('idle') is True
                and re.search(r'No process in device\.|No running processes found',row.get('output',''),re.I)
                for row in processes.values()))
    checks = dict(pre_verifier_zero=receipt.get('pre_verifier',{}).get('exit_code') == 0,
        post_verifier_zero=receipt.get('post_verifier',{}).get('exit_code') == 0,
        owned_processes_cleared=receipt.get('owned_remaining') == [],
        physical_cards=receipt.get('physical_cards') == [2,5,6,7],
        hardware_idle_before=actual_hardware_idle(receipt.get('devices_before')),
        hardware_idle_after=actual_hardware_idle(receipt.get('devices_after')),
        experimental_sources_unchanged=receipt.get('experimental_sources_unchanged') is True,
        external_sources_unchanged=receipt.get('external_sources_unchanged') is True)
    exit_path = directory/'worker-exit.json'
    worker_exit = read_json(exit_path) if exit_path.exists() else None
    checks['original_workers_closed'] = bool(worker_exit and worker_exit.get('closed') is True
                                            and worker_exit.get('all_exited') is True)
    checks['original_child_exit_zero']=receipt.get('child',{}).get('exit_code')==0
    completed_path=directory/'completed.json'
    completed=read_json(completed_path) if completed_path.exists() else None
    checks['original_completion_replies']=bool(completed and completed.get('action')==receipt['action']
        and completed.get('replies')==[['calibrated',rank,dict(rows=0,timings=0)] for rank in range(4)])
    for key,value in checks.items():
        if not value:errors.append(key)
    return dict(passed=not errors, checks=checks, failed_checks=errors,
        actual_child=receipt.get('child'), original_worker_exit=worker_exit,
        original_completed_artifact=descriptor(completed_path) if completed else None,
        devices_before=receipt.get('devices_before'),devices_after=receipt.get('devices_after'),
        pre_verifier=receipt.get('pre_verifier'),post_verifier=receipt.get('post_verifier'),
        owned_remaining=receipt.get('owned_remaining'),
        actual_original_status=receipt['status'],actual_original_errors=receipt.get('errors'))


def trace_observation(trace):
    allocator=trace['allocator'];peak=trace['active_peak']
    require(type(peak) is int and 0<peak<=64
            and peak==trace['decode_counters']['spec_rhythm_peak_active_requests']
            and trace['actual_capacity64_scope_completed'] is (peak>32),
            'Configured64 cannot substitute for actual active>32 observation')
    require(allocator['pool_blocks']==544 and type(allocator['referenced_page_highwater']) is int
            and 0<allocator['referenced_page_highwater']<=544 and allocator['final_referenced_pages']==0
            and allocator['acquired_pages']==allocator['released_pages']
            and type(allocator['acquired_pages']) is int and allocator['acquired_pages']>0
            and allocator['allocation_storage_rows']==[527], 'Actual page/storage highwater/release arithmetic differs')
    require(type(trace['source_generation']) is int and type(trace['source_generation_before']) is int
            and trace['source_generation']==trace['source_generation_before']+1,
            'Actual fresh source epoch did not advance exactly once')
    return dict(active_peak=peak,actual_capacity64_scope_completed=peak>32,allocator=allocator,
        native=trace['native'],decode_counters=trace['decode_counters'],phase_seconds=trace['phase_seconds'],
        normal_target_N_distribution=dict(Counter(c['N'] for c in trace['normal_target_calls'])),
        normal_target_M_S_distribution=dict(Counter(f"{c['physical_M']}/{c['physical_S']}" for c in trace['normal_target_calls'])),
        standalone_prefill_calls=len(trace['standalone_prefill_calls']),
        logical_event_kinds=dict(Counter(e['kind'] for e in trace['logical_events'])),
        source_generation=trace['source_generation'],source_generation_before=trace['source_generation_before'],
        sealed_before=trace['sealed_before'],sealed_after=trace['sealed_after'],cache_released=trace['cache_released'])


def analyze(tag, source=SOURCE):
    source = Path(source).resolve()
    directory = source/'local-output'/tag
    receipt_path = source/f'local-receipt-{tag}.json'
    receipt = read_json(receipt_path)
    require(receipt.get('schema') == 'V96.capacity64-runtime-supervisor.v1'
            and receipt.get('status') in ('complete','failed'), 'Wait for actual terminal V96 supervisor receipt')
    result = dict(schema='ROOT.V96.actual-capacity64-independent-audit.v1',tag=tag,
        UTC=datetime.now(timezone.utc).isoformat(),original_status=receipt['status'],
        original_errors=receipt.get('errors'), original_receipt=descriptor(receipt_path),
        execution=dict(argv=list(sys.argv),executable=sys.executable,python_version=sys.version,
            dont_write_bytecode=sys.dont_write_bytecode,model_or_NPU_imports_allowed=False,
            environment_wrapper_proven_by_argv=False,
            boundary='Read-only HOST standard-library archive/JSON arithmetic plus frozen HOST validators; no device query/initialization'),
        errors=[], artifacts=[], partial_evidence={}, full_qualification=False,
        formal_Goodput=False,online_qualified=False,whole_workload_AR_equality_qualified=False)
    def section(name, callback, *, retain=True):
        try:
            value = callback()
            result[name] = value if retain else dict(actual_file_read_and_checked=True)
            return value
        except Exception as error:
            result['errors'].append(dict(section=name,error=type(error).__name__+': '+str(error),
                                         traceback=traceback.format_exc()))
            result[name] = dict(passed=False,error=str(error))
            return None
    sources = section('source_CPU_archive_and_environment',lambda:audit_sources(source,receipt))
    section('cleanup',lambda:audit_cleanup(receipt,directory))
    reports, traces, chains = {}, {role:{} for role in ROLES}, {}
    for rank in range(4):
        path = directory/f'rank{rank}-local-diagnostic.json'
        if path.exists():
            value = section(f'worker_report_rank{rank}',lambda path=path:read_json(path),retain=False)
            if value is not None:
                reports[rank] = value
                result['artifacts'].append(dict(kind='original-worker-report',**descriptor(path)))
        for role in ROLES:
            path = directory/f'rank{rank}-{role}-actual-trace.json.gz'
            if not path.exists():continue
            outer = next((t for t in reports.get(rank,{}).get('traces',[]) if t['role'] == role),None)
            def load(path=path,outer=outer):
                if outer is not None:
                    artifact = outer['artifact'];value=read_gzip(artifact,directory)
                else:artifact,value=recover_gzip(path,directory)
                result['artifacts'].append(dict(kind='actual-trace',rank=rank,role=role,
                    descriptor_bound_to_original_worker=outer is not None,**artifact))
                return value
            value = section(f'load_trace_rank{rank}_{role}',load,retain=False)
            if value is not None:traces[role][rank]=value
        path = directory/f'rank{rank}-same-M-qualification.json.gz'
        if path.exists():
            def load_chain(path=path,rank=rank):
                outer=reports.get(rank,{}).get('mixed_same_M_qualification')
                if outer is not None:artifact=outer['artifact'];value=read_gzip(artifact,directory)
                else:artifact,value=recover_gzip(path,directory)
                result['artifacts'].append(dict(kind='actual-same-M',rank=rank,
                    descriptor_bound_to_original_worker=outer is not None,**artifact))
                return value
            value=section(f'load_same_M_rank{rank}',load_chain,retain=False)
            if value is not None:chains[rank]=value
    result['partial_evidence']=dict(original_worker_reports=sorted(reports),
        same_M_ranks=sorted(chains),trace_ranks={role:sorted(rows) for role,rows in traces.items()})
    if sources:
        section('production_TPOT_source_formula',lambda:production_tpot_contract(source))
        with frozen_host_modules(source) as modules:
            section('actual_startup_inventory',lambda:audit_startup(reports,modules,source))
            section('same_M_numerical_and_transaction_qualification',lambda:audit_same_m(chains,reports,modules))
            section('actual_inventory_progress',lambda:audit_progress(source,modules))
            workload=modules['audit_capacity64_v96']
            per_roles=[]
            for role,records in traces.items():
                if not records:continue
                count=463 if role=='full463' else 65
                rows,prefix=workload.manifest_rows(MANIFEST,count)
                if set(records)==set(range(4)):
                    item=section('actual_workload_'+role,lambda records=records,rows=rows,prefix=prefix,role=role:
                        workload.audit_trace_set([records[r] for r in range(4)],rows,prefix,zero=role!='warmup65'))
                    if item is not None:per_roles.append(dict(role=role,**item))
                else:
                    result['errors'].append(dict(section='actual_workload_'+role,error='Incomplete actual four-rank trace set'))
                if 1 in records:
                    section('independent_metrics_'+role,lambda trace=records[1],rows=rows:metric_rows(trace,rows))
            if len(reports)==4:
                section('original_ROOT_workload_validator',lambda:workload.audit_worker_reports(
                    [reports[r] for r in range(4)],MANIFEST,directory))
            else:result['errors'].append(dict(section='original_worker_completeness',error='Four original final worker reports absent; recovered artifacts do not replace them'))
            result['trace_observations']={}
            for role,records in traces.items():
                if not records:continue
                result['trace_observations'][role]={}
                for rank,trace in records.items():
                    observation=section(f'host_capacity_page_epoch_{role}_rank{rank}',lambda trace=trace:trace_observation(trace))
                    result['trace_observations'][role][str(rank)]=observation
            for rank in range(4):
                available=[traces[role][rank] for role in ROLES if rank in traces[role]]
                if any(after['source_generation_before']!=before['source_generation'] for before,after in zip(available,available[1:])):
                    result['errors'].append(dict(section='source_epoch_chain',error='Actual fresh generations do not form consecutive scopes for rank'+str(rank)))
            actual_files=[path for path in source.rglob('*') if path.is_file()]
            total=sum(path.stat().st_size for path in actual_files)
            result['actual_private_storage']=dict(bytes=total,limit_bytes=256*1024**2,
                maximum_file_bytes=max(path.stat().st_size for path in actual_files),file_limit_bytes=MAX_COMPRESSED)
            if total>=256*1024**2 or any(path.stat().st_size>MAX_COMPRESSED for path in actual_files):
                result['errors'].append(dict(section='actual_private_storage',error='Actual V96 private/file storage bound exceeded'))
    result['original_complete_full463']=bool(receipt['status']=='complete' and len(reports)==4
        and all(report.get('full463_completed') is True for report in reports.values()))
    result['actual_capacity64_scope_exercised']=any(t['active_peak']>32 for records in traces.values() for t in records.values())
    result['actual_full463_workload_artifact_audit_passed']=bool(result.get('actual_workload_full463')
        and 'error' not in result['actual_workload_full463'])
    result['actual_local_same_M_numerical_audit_passed']=result.get('same_M_numerical_and_transaction_qualification',{}).get('passed') is True
    result['full_qualification']=bool(result['original_complete_full463'] and not result['errors']
        and result['actual_capacity64_scope_exercised']
        and result.get('cleanup',{}).get('passed') is True
        and result.get('actual_inventory_progress',{}).get('passed') is True
        and result.get('same_M_numerical_and_transaction_qualification',{}).get('passed') is True)
    result['qualification_scope']='Actual full463 workload and source-bound local five-phase same-M qualification, source/CPU/archive/environment/cleanup; never all-request independent AR equality or general online qualification'
    # Keep original failed/completion status. A useful partial numerical result
    # cannot rewrite an actual worker/supervisor failure into a successful run.
    result['all_readable_evidence_checks_passed']=not result['errors'] and all(
        result.get(name,{}).get('passed') is True for name in
        ('source_CPU_archive_and_environment','cleanup','actual_startup_inventory',
         'same_M_numerical_and_transaction_qualification'))
    result['self']=descriptor(Path(__file__))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.tag),'Invalid tag')
    require(sys.dont_write_bytecode,'Use retained Python3.11 -B for HOST audit')
    output=args.output or BASE/'budget-diagnostics'/f'ROOT_V96_ACTUAL_CAPACITY64_{args.tag}_AUDIT.json'
    require(not output.exists(),'Refusing to replace an actual audit attempt')
    result=analyze(args.tag)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    output.chmod(0o600)
    full=result.get('independent_metrics_full463',{})
    print(json.dumps(dict(output=str(output),sha256=file_sha(output),original_status=result['original_status'],
        errors=len(result['errors']),original_complete_full463=result['original_complete_full463'],
        full_qualification=result['full_qualification'],full463_requests=full.get('requests'),
        paper_goodput=full.get('paper_goodput_tokens_per_second'),raw_tokens_per_second=full.get('raw_tokens_per_second'),
        formal_Goodput=False)))
    return int(not result['full_qualification'])


if __name__=='__main__':
    raise SystemExit(main())
