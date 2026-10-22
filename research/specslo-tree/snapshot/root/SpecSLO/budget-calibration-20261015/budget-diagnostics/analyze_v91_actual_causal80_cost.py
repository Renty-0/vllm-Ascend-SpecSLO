#!/usr/bin/env python3
"""Read-only V91 actual-result audit. Never imports torch or launches a process.

Run only after the supervisor has completed:
  python -B analyze_v91_actual_causal80_cost.py --tag causal80-three-case-cost-v1
Writes two external diagnostic JSON files after every gate passes. Running,
failed, incomplete or stale evidence produces a refusal and no result file.
The four ranks observe 15 synchronized calls, not 60 independent samples.
"""
import argparse
import ast
from collections import Counter
import datetime
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tarfile
import textwrap

sys.dont_write_bytecode = True
BASE = Path('/root/SpecSLO/budget-calibration-20261015')
WORK = BASE/'tree-causal80-cost-v91'
OUT = BASE/'budget-diagnostics'
FREEZE = 'CAUSAL80_COST_SOURCE_MANIFEST_V91.json'
FREEZE_SHA = '55fe98e3d6445621e18c42fd87e4be6a6b2276b161922a6fa4e1b0e543fb9591'
PRE_AUDIT = OUT/'ROOT_V91_PRE_NPU_SOURCE_AND_HOST_AUDIT.json'
PRE_AUDIT_SHA = '192c94a3ae441ced0a4af980dd70f3488fd2951d031b9d4be83fb2538f869f9f'
NAMES = ('B4_parent128', 'B4_causal80', 'B5_full96')
MC2_FIELDS = ('fused_attempt','fused_success','fallback','exception','full_fused_attempt',
    'full_fused_success','native_epilogue_attempt','native_epilogue_success',
    'native_epilogue_chained_attempt','native_epilogue_chained_success','native_epilogue_chain_flush')
MC2_ROWS = [40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,116,120,124,128,165,197]
OPERATOR = 'tp3_native_matmul+allreduce_add_rmsnorm'
PROFILE_SHA = 'ad9e0648810ed697679ee84b8bbc2939ea14bfb687d0215d27f403ad5526db1c'
PHASES = ('draft_first_whole_method_ms','target_whole_forward_ms',
    'original_verdict_broadcast_ms','original_commit_protocol_ms',
    'next_draft_whole_method_including_repair_ms')


def require(ok, label):
    if not ok:
        raise RuntimeError('V91 actual external audit refused: '+label)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def true_fields(row, fields, label):
    require(isinstance(row,dict) and all(row.get(k) is True for k in fields), label)


def numeric(value, label):
    require(type(value) in (int,float) and math.isfinite(value) and value >= 0, label)
    return value


def stats(values):
    values = list(values)
    require(bool(values), 'nonempty timing sample set')
    for v in values:
        numeric(v, 'finite timing')
    return dict(samples_ms=values,median_ms=statistics.median(values),
                min_ms=min(values),max_ms=max(values))


def host_module(name):
    # Only source-pinned stdlib-only modules are executed by this parser.
    spec = importlib.util.spec_from_file_location(name,WORK/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    require('torch' not in sys.modules, 'external HOST parser imported torch')
    return module


class ObjectPairs(list):
    """Retain duplicate object keys in the raw runtime fusion diagnostic."""


def fusion_runtime_output(path, *, started_unix, completed_unix):
    path = Path(path)
    require(path.name == 'fusion_result.json' and path.is_file() and not path.is_symlink(),
            'only exact regular fusion_result.json is an allowed runtime diagnostic')
    raw = path.read_bytes()
    stat = path.stat()
    require(0 < len(raw) <= 64*1024 and started_unix <= stat.st_mtime <= completed_unix,
            'bounded fusion diagnostic observed in actual run interval')
    value = json.loads(raw,object_pairs_hook=ObjectPairs)
    require(isinstance(value,ObjectPairs) and 1 <= len(value) <= 128,
            'fusion root is a bounded JSON object (duplicate keys retained)')
    allowed_passes = {'TransdataCastFusionPass','TransdataFz2FzgFusionPass','TransdataFzg2FzFusionPass'}
    observations = []
    for key,session in value:
        require(isinstance(key,str) and re.fullmatch(r'session_and_graph_id_\d+_\d+',key),
                'known fusion session/graph structural key')
        require(isinstance(session,ObjectPairs) and len(session) == 1 and session[0][0] == 'graph_fusion',
                'known graph_fusion-only structure')
        passes = session[0][1]
        require(isinstance(passes,ObjectPairs) and len(passes) == 3
            and {name for name,counts in passes} == allowed_passes, 'only known Transdata fusion pass records')
        summary = {}
        for name,counts in passes:
            require(isinstance(counts,ObjectPairs) and len(counts) == 2
                and {k for k,v in counts} == {'effect_times','match_times'}
                and all(isinstance(v,str) and v.isascii() and v.isdigit() for k,v in counts),
                'fusion effect/match are explicit nonnegative decimal string counts')
            summary[name] = dict(counts)
        observations.append(dict(session_and_graph_id=key,graph_fusion=summary))
    require(path.read_bytes() == raw and path.stat().st_mtime_ns == stat.st_mtime_ns,
            'fusion diagnostic unchanged while reading')
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
        mtime_ns=stat.st_mtime_ns,mtime_UTC=datetime.datetime.fromtimestamp(stat.st_mtime,datetime.timezone.utc).isoformat(),
        object_pair_records=len(observations),unique_session_graph_keys=len({r['session_and_graph_id'] for r in observations}),
        duplicate_object_keys_retained=True,records=observations,
        classified_as='known-runtime-output-by-exact-name-and-structure',
        writer_identity_qualified=False,writer='unattributed',consumed_source_input=False)


def postrun_top_level(work, bound_names, *, started_unix, completed_unix):
    """Do not re-run launch collect after device libraries produced outputs.

    Every declared static input was already hashed against its frozen map.
    Only the exact structured runtime fusion JSON and named cost receipts may
    be additional JSON/code inputs. Unknown static/code files still fail.
    """
    outputs = []
    for path in Path(work).iterdir():
        if not path.is_file() or path.name in bound_names:
            continue
        if path.name == 'fusion_result.json':
            outputs.append(fusion_runtime_output(path,started_unix=started_unix,completed_unix=completed_unix))
        elif re.fullmatch(r'cost-receipt-causal80-three-case-cost-[A-Za-z0-9-]+\.json',path.name):
            continue  # The selected actual receipt is independently audited.
        elif path.suffix in ('.py','.sh','.json','.pyc','.so','.yaml','.yml','.toml','.ini'):
            require(False,'unexpected unbound postrun code/static input: '+path.name)
    return outputs


def postrun_readiness(receipt,freeze):
    """Recheck immutable HOST/freeze/parents, binding actual pre-launch proof."""
    require(sha(PRE_AUDIT) == PRE_AUDIT_SHA, 'root actual pre-NPU audit identity')
    pre = read(PRE_AUDIT)
    cpu = read(WORK/'CPU_RESULT_V91.json')
    module = host_module('causal80_cost_readiness_v91')
    static = {k:v for k,v in freeze['files'].items() if k != 'CPU_RESULT_V91.json'}
    require(pre['all_checks_passed'] is True and pre['checks'] and all(v is True for v in pre['checks'].values())
        and pre['CPU_sha256'] == sha(WORK/'CPU_RESULT_V91.json')
        and pre['freeze_sha256'] == FREEZE_SHA
        and pre['HOST_tests'] == cpu['tests_run']
        and pre['local_sources'] == len(static)
        and pre['external_inputs'] == len(freeze['external_files']), 'actual pre-NPU source/HOST proof binding')
    require(cpu['schema'] == 'v91-causal80-focused-HOST-v1' and cpu['status'] == 'passed'
        and cpu['modules'] == list(module.CPU_MODULES) and cpu['tests_run'] >= 30
        and cpu['failures'] == cpu['errors'] == [] and cpu['NPU_initialized'] is False
        and cpu['model_initialization_attempts'] == cpu['NPU_initialization_attempts'] == []
        and cpu['input_hashes'] == cpu['input_hashes_after'] == static
        and cpu['external_input_hashes'] == cpu['external_input_hashes_after'] == freeze['external_files'],
        'unchanged full actual HOST evidence and exact source/environment closure')
    require(sha(pre['source_archive']) == pre['source_archive_sha256'], 'actual pre-NPU HOST source archive')
    readiness = pre['readiness']
    require(readiness == receipt['prerequisites']
        and readiness['source_contract'] == module.parent_contract()
        and readiness['final_source_manifest_sha256'] == FREEZE_SHA
        and readiness['actual_HOST_tests'] == cpu['tests_run']
        and readiness['local_source_frozen'] is True
        and readiness['NPU_qualified'] is readiness['policy_modified'] is False,
        'same actual launch readiness plus freshly rechecked parent contract')
    require(pre['created_UTC'] < receipt['started_UTC'], 'root readiness predates actual device run')
    return readiness


def static_mc2(row, rank):
    true_fields(row, ('passed',), 'MC2 full static passed')
    require(row['actual_family_rows'] == [80,96,128] and row['native_flags_modified'] is False,
            'MC2 three physical M and unchanged flags')
    inv = row['original_inventory']
    true_fields(inv, ('local_passed','profile_standalone_chained_flush'), 'original MC2 inventory')
    require(inv['rank'] == rank and inv['role'] == 'target'
        and inv['profile_disk_sha256'] == PROFILE_SHA
        and inv['profile_operator'] == OPERATOR
        and inv['native_qualification_flags_modified'] is False
        and inv['manifest_route_count'] == 128
        and inv['planned_full_chain_rows'] == MC2_ROWS
        and inv['model_chained_row_counts'] == MC2_ROWS
        and all(v is True for v in inv['checks'].values())
        and all(inv[k] == [] for k in ('missing_chained_rows','unexpected_chained_rows',
            'disabled_routes','routes_with_missing_or_unqualified_rows','failed_checks')),
        'actual 128-route/22-row original MC2 inventory')
    routes = inv['route_details']
    require(len(routes) == 128 and {(r['layer_index'],r['projection_kind']) for r in routes}
        == {(layer,kind) for layer in range(64) for kind in ('attention','down')},
        'all64 attention/down route identities')
    require(all(r['enabled'] is True and r['operator_kind'] == OPERATOR
        and r['planned_decisions_present'] == 22 and r['missing_rows'] == []
        and r['unqualified_rows'] == [] for r in routes), 'all MC2 route decisions')


def raw_mc2(row, rank):
    true_fields(row, ('passed',), 'raw MC2 passed')
    static_mc2(row['static_contract'],rank)
    dispatch = row['dispatch']
    true_fields(dispatch, ('passed',), 'raw MC2 dispatch')
    before,after = dispatch['before'],dispatch['after']
    require(set(before) == set(after) == set(MC2_FIELDS), 'eleven original MC2 counters')
    require(all(type(v) is int and v >= 0 for snap in (before,after) for v in snap.values()),
            'MC2 nonnegative integer snapshots')
    delta = {k:after[k]-before[k] for k in before}
    require(delta == dispatch['values'] and all(v >= 0 for v in delta.values())
        and delta['fallback'] == delta['exception'] == 0, 'raw MC2 counter subtraction')
    require(type(dispatch['native_success_required']) is bool, 'explicit MC2 dispatch kind')
    if dispatch['native_success_required']:
        require(delta['native_epilogue_success'] > 0, 'cold/reference positive MC2 dispatch')
    if 'checks' in row:
        require(row['checks'] and all(v is True for v in row['checks'].values()), 'MC2 actual entry checks')
        for entry in row['after_entries'].values():
            true_fields(entry, ('exists','valid'), 'MC2 resident original entry')
            require(entry['checks'] and all(v is True for v in entry['checks'].values()),
                    'MC2 all64 task geometry/handles/events')
        mode = 'cold_capture' if row['history']['cold_capture_required'] else 'existing_replay'
    else:
        require(dispatch['native_success_required'] is True, 'independent eager positive dispatch required')
        mode = 'independent_eager'
    return dict(mode=mode,positive_native_dispatch=delta['native_epilogue_success'],
                fallback=delta['fallback'],exception=delta['exception'])


def protocol_proof(row, wanted_hashes, case):
    p = row['Native_protocol']
    true_fields(p, ('extracted_structural_and_numerical_lanes_unchanged',
        'finite_preflight_cpu_rendezvous_and_native_compaction',
        'actual_observed_target_verdict_only','all16_requests_retained'), 'original Native protocol')
    require((p['original_decode_sha256'],p['canonical_decode_sha256']) == wanted_hashes
        and p['per_request_grants_preserved'] == case['budgets'], 'Native source/grants')


def commit_proof(p):
    true_fields(p, ('all_real_rows','all_layers','independent_source_KV',
        'perrow_original_query_offset','all64_target_layers'), 'actual target compaction KV proof')
    require(p['dense_AR_qualified'] is False, 'local commit must not claim dense AR')


def native_witnesses(records):
    require(records, 'actual first/next Native witness records')
    final = {}
    for row in records:
        true_fields(row, ('qualification_only','state_before_is_after_actual_original_call',
            'unpublished_alternate_input'), 'original unpublished Native witness')
        require(row['native_flags_assigned'] is False and row['key'] == row['actual_key']
            and row['owner_count'] == 16 and row['actual_execution']['used_aclgraph'] is True
            and row['actual_execution']['mode'] in ('capture_replay','replay'), 'actual original Native call')
        final[(row['kind'],row['key'])] = row
    require(any(kind == 'draft' for kind,key in final), 'actual draft-tail witness')
    for (kind,key),row in final.items():
        true_fields(row, ('passed',), 'final Native witness must pass')
        if row.get('already_originally_qualified') is True:
            require(row['state_before']['runtime_validated'] is True, 'read actual resident Native qualification')
            counters = row['counters_observed']
            require(all(counters[k] == 0 for k in ('runtime_validation_failures',
                'eager_fallback_calls','disabled_entry_calls','unclassified_calls')), 'resident Native failure counters')
        else:
            true_fields(row, ('alternate_used','same_key_and_object','all28_queryKV_original_matcher',
                'non_query_known_prefix_bitwise','readonly_PAD_prefix0_protected_by_real_owner0_prefix',
                'ledger_unchanged','original_actual_inputs_restored','restored_output_native_match',
                'restored_queryKV_native_match','restored_non_query_prefix_bitwise','restored_ledger'),
                'changed-input original guard and actual restoration')
            require(row['counters_delta']['runtime_validation_calls'] == 1
                and row['counters_delta']['changed_input_validation_calls'] == 1
                and all(row['counters_delta'][k] == 0 for k in ('runtime_validation_failures',
                    'eager_fallback_calls','disabled_entry_calls','unclassified_calls')),
                'actual one changed-input Native validation')
        if kind == 'target':
            require(row['native_valid_row_count'] is None
                and row['target_hidden_real_rows_are_not_a_native_validation_field'] is True,
                'hidden real-row witness is actual metadata, not an assigned Native flag')


def observed_signature(row):
    return {k:row[k] for k in ('g','accepted','emitted','selected_original_nodes','plain_path_signature')}


def signature_differences(a_name, b_name, a, b):
    differences, lengths, grants = [],[],[]
    budgets = {'B4_parent128':4,'B4_causal80':4,'B5_full96':5}
    for owner,(aa,bb) in enumerate(zip(a['emitted'],b['emitted'])):
        av,bv = [x for x in aa if x >= 0],[x for x in bb if x >= 0]
        positions = [i for i in range(min(len(av),len(bv))) if av[i] != bv[i]]
        if positions:
            differences.append(dict(owner=owner,positions=positions,first_tokens=av,second_tokens=bv))
        if len(av) != len(bv):
            lengths.append(dict(owner=owner,first=len(av),second=len(bv)))
        if budgets[a_name] == budgets[b_name] and a['g'][owner] != b['g'][owner]:
            grants.append(dict(owner=owner,B=budgets[a_name],first_g=a['g'][owner],second_g=b['g'][owner]))
    return dict(first=a_name,second=b_name,common_valid_position_token_mismatches=differences,
        emitted_length_differences=lengths,same_grant_g_label_differences=grants,
        original_raw_rows_retained_in_cost_records=True,same_target_tokens_required=False,
        pooling_into_gfit_qualified=False)


def source_protocol_hashes(external):
    matches = [Path(p) for p in external if Path(p).name == 'native_engine.py'
               and 'vllm_ascend/spec_decode/pearl' in p]
    require(len(matches) == 1, 'unique original Native source in closure')
    source = matches[0].read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'NativePearlEngine')
    func = next(n for n in cls.body if isinstance(n,ast.FunctionDef)
                and n.name == '_generate_spec_rhythm_tree_decode')
    lines = source.splitlines(keepends=True)
    start = min([func.lineno]+[d.lineno for d in func.decorator_list])
    original = textwrap.dedent(''.join(lines[start-1:func.end_lineno]))
    # Match inspect.getsource, including its decorator, without importing Native.
    # Transport's module imports torch; execute only its two exact HOST AST
    # definitions. Their bodies manipulate Python syntax, never model state.
    canonical = original
    for name,cls_name in (('canonical_eager','DecodeRewrite'),('host_tree_transport','Rewrite')):
        parsed = ast.parse((WORK/(name+'.py')).read_text())
        definitions = [n for n in parsed.body if
            isinstance(n,ast.ClassDef) and n.name == cls_name or
            isinstance(n,ast.FunctionDef) and n.name == 'patch_decode_source']
        require(len(definitions) == 2, 'exact source-only protocol rewrite definitions')
        namespace = dict(ast=ast,Counter=Counter)
        exec(compile(ast.Module(body=definitions,type_ignores=[]),str(WORK/(name+'.py'))+':HOST-AST','exec'),namespace)
        canonical = namespace['patch_decode_source'](canonical)
    return (hashlib.sha256(original.encode()).hexdigest(),hashlib.sha256(canonical.encode()).hexdigest())


def audit(tag):
    require(tag.startswith('causal80-three-case-cost-') and tag.replace('-','').isalnum(), 'bounded tag')
    receipt_path = WORK/('cost-receipt-'+tag+'.json')
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    require(receipt.get('schema') == 'v91-causal80-cost-receipt-v1'
        and receipt.get('status') == 'complete', 'terminal complete receipt required before any actual output')
    require(receipt['errors'] == [] and receipt['physical_cards'] == [2,5,6,7]
        and receipt['child']['exit_code'] == 0
        and receipt['pre_verifier']['exit_code'] == receipt['post_verifier']['exit_code'] == 0,
        'successful child and original pre/post verifier')
    require(receipt['devices_before']['all_idle'] is True and receipt['devices_after']['all_idle'] is True
        and receipt['owned_remaining'] == [], 'idle cards and owned-process cleanup')
    true_fields(receipt, ('experimental_sources_unchanged','external_sources_unchanged',
        'local_scope_complete'), 'complete source closure and local scope')
    require(receipt['formal_Goodput'] is False and receipt['production_overlap_qualified'] is False
        and receipt['whole_workload_AR_equality_qualified'] is False, 'local-only receipt')
    require(0 <= numeric(receipt['elapsed_seconds'],'bounded elapsed') < 2400
        and 0 <= numeric(receipt['private_bytes'],'bounded private storage') < 256*1024**2,
        'original supervisor resource bounds')
    require(sha(WORK/FREEZE) == FREEZE_SHA, 'reviewed V91 exact source freeze')
    freeze = read(WORK/FREEZE)
    expected_local = {str(WORK/name):digest for name,digest in freeze['files'].items()}
    expected_local[str(WORK/FREEZE)] = FREEZE_SHA
    require(receipt['input_hashes'] == expected_local
        and receipt['external_input_hashes'] == freeze['external_files'], 'run closure exactly matches freeze')
    for p,digest in {**expected_local,**freeze['external_files']}.items():
        require(sha(p) == digest, 'unchanged current source/input: '+p)
    archive_path = Path(receipt['source_snapshot'])
    require(archive_path.parent == WORK/'source-snapshots'
        and sha(archive_path) == receipt['source_snapshot_sha256'], 'actual runtime source archive hash')
    archive_hashes = {}
    with tarfile.open(archive_path,'r:gz') as archive:
        for item in archive.getmembers():
            require(item.isfile() and Path(item.name).name == item.name and item.name not in archive_hashes,
                    'regular unique flat archived source')
            archive_hashes[str(WORK/item.name)] = hashlib.sha256(archive.extractfile(item).read()).hexdigest()
    require(archive_hashes == expected_local, 'every runtime archive member matches source closure')
    sys.path.insert(0,str(WORK))
    readiness = postrun_readiness(receipt,freeze)
    require(readiness == receipt['prerequisites'], 'unchanged readiness from actual start to final audit')
    started = datetime.datetime.fromisoformat(receipt['started_UTC']).timestamp()
    runtime_outputs = postrun_top_level(WORK,set(freeze['files'])|{FREEZE},
        started_unix=started,completed_unix=receipt_path.stat().st_mtime)
    directory = WORK/'context-cost-sweep-v1'/tag
    require(read(directory/'prerequisites.json') == readiness, 'child actual readiness binding')
    paths = [directory/f'rank{rank}-four-family-cost.json' for rank in range(4)]
    require(receipt['bounded_local_report_paths'] == [str(p) for p in paths], 'exact four actual report paths')
    reports = [read(p) for p in paths]
    completed = read(directory/'completed.json')
    frozen_audit = host_module('causal80_cost_report_audit_v91').audit_reports(reports,completed)
    require(receipt['actual_report_audit'] == frozen_audit
        and receipt['completed_replies'] == completed['replies'], 'recomputed original report audit')
    cfg = read(directory/'config.json')
    for k,value in dict(draft_tensor_parallel_size=1,target_tensor_parallel_size=3,
        max_num_seqs=32,num_kvcache_blocks=544,max_model_len=4096,max_aclgraph_entries=512,
        spec_rhythm_tree_width=2,spec_rhythm_tree_depth=4,spec_rhythm_verification_budget=96,
        target_use_paged_attention=True,enable_mc2=True).items():
        require(cfg[k] == value, 'actual config '+k)
    require(cfg['mc2_profile'] == '/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/mc2-profile.json',
            'actual frozen MC2 profile path')
    scope = host_module('four_family_scope')
    wanted_protocol = source_protocol_hashes(freeze['external_files'])
    source_data = scope.fixture()
    identity = hashlib.sha256(json.dumps(dict(contexts=source_data['contexts'],
        repairs=source_data['initial_missing_suffix_lengths']),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    changed_contexts = json.loads(json.dumps(source_data['contexts']))
    for context in changed_contexts:
        context['token_ids'][-1] = (context['token_ids'][-1]+1)%151936
    changed_identity = hashlib.sha256(json.dumps(dict(contexts=changed_contexts,
        repairs=source_data['initial_missing_suffix_lengths']),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    prefixes = [len(c['token_ids'])-1 for c in source_data['contexts']]
    require(frozen_audit['source_identity_sha256'] == identity
        and frozen_audit['source_prefix_lengths'] == prefixes, 'actual untouched original16 source histories')
    raw_evidence = {str(p):sha(p) for p in [receipt_path,*paths,directory/'completed.json',
        directory/'config.json',directory/'prerequisites.json',WORK/FREEZE,PRE_AUDIT]}
    raw_evidence.update({row['path']:row['sha256'] for row in runtime_outputs})
    preflight = read(directory/'rank0-preflight.json')
    require(len(preflight) == 4, 'all4 actual preflight memory/static observations')
    for rank in range(4):
        pre_path = directory/f'rank{rank}-preflight.json'
        require(read(pre_path) == preflight, 'all ranks retain the same preflight all-gather')
        raw_evidence[str(pre_path)] = sha(pre_path)
        m = preflight[rank]['memory']
        require(m['blocks'] == 544 and 60*1024**3 <= m['total'] <= 64*1024**3
            and m['free'] >= 6*1024**3 and m['reserved'] <= 54*1024**3
            and m['peak'] <= 54*1024**3 and m['entries'] <= 512,
            'persisted actual preflight resource guard')
        if rank:
            static_mc2(preflight[rank]['MC2'],rank)
    witnessed = []
    for rank,report in enumerate(reports):
        require(report['declared_numerical_records'] == 9 and report['declared_methods'] == 3
            and report['samples_per_method'] == 5 and report['timing'] == report['cost'],
            'three methods/five repeats/original timing rows')
        true_fields(report, ('normal_only','preselected_budget_vectors_not_a_new_allocator',
            'local_cost_only','qualified_sequential_local_scope_only',
            'original_root_diagnostics_retained_in_cost','original_tail_observer_retained_in_cost',
            'no_extra_scoped_observer_in_cost'), 'original cost observer/scope contract')
        require(report['cross_M_pooling_into_gfit_qualified'] is False, 'no cross-M gain pooling')
        qualified = {}
        for q in report['qualification']:
            case = q['case']
            require(q['source_identity_sha256'] == (changed_identity if q['changed_input_variant'] else identity)
                and q['source_prefix_lengths'] == prefixes
                and q['initial_missing_suffix_lengths'] == [1]*8+[0]*8,
                'F/T/F own fresh original/changed/restored source tokens')
            protocol_proof(q,wanted_protocol,case)
            true_fields(q['whole_method_restoration'], ('passed','no_extra_scoped_observer',
                'actual_method_freshly_reexecuted','first_and_next_public_exact_confidence',
                'first_and_next_ledger','all28_draft_query_KV','all64_target_physical_query_KV',
                'all_source_prefixes_protected','actual_observed_verdict_unchanged',
                'comparison_only_snapshots_never_copied_to_candidate',
                'independent_reference_and_bridge_outside_all_cost_timing',
                'original_root_retention_and_permanent_tail_observer_unchanged'), 'whole-method plain restoration')
            if rank == 0:
                for label in ('first','next'):
                    rows = q['original_same_key_native_witnesses'][label]
                    native_witnesses(rows)
                    witnessed.extend(rows)
                d = q['draft_oracle']
                true_fields(d, ('observed_verdict_only','independent_first_history_replayed',
                    'no_dense_newprefix_prefill','all_original_source_prefixes_protected',
                    'each_method_valid_prefix_before_missing_suffix_bitwise_protected'), 'draft independent history')
                require(d['candidate_reference_known_prefix_cross_bitwise_equality_required'] is False,
                        'same-method prefix protection, no invented cross reference bitwise equality')
                for label in ('first','next'):
                    true_fields(d[label], ('passed','exact_top2_confidence','all_layer_query_KV',
                        'independent_perrow_ledger','no_candidate_KV_or_ledger_reference',
                        'all_actual28_draft_layer_query_KV','all_physical_first_and_tail_hidden',
                        'exact_fullphysical_encoded','exact_aggregate_confidence',
                        'original_V51_independent_history','reference_never_initialized_from_candidate_KV'),
                        'all28 draft '+label+' true independent query KV')
                    require(type(d[label]['native_validated_real_row_count']) is int
                        and d[label]['native_validated_real_row_count'] >= 16, 'real16 draft Native gate')
            else:
                t = q['target_oracle']
                true_fields(t, ('passed','all_physical_hidden','all64_target_layer_query_KV',
                    'exact_greedy_tokens','source_prefix_protected','whole_target_restored',
                    'original_hidden_changed_input_gate','original_greedy_native_gate',
                    'no_candidate_KV_used_to_build_reference'), 'all64 target true independent same-M query KV')
                require((t['M'],t['S'],t['real_Q'],t['family']) ==
                    (case['M'],case['S'],case['Q'],case['family'])
                    and t['actual_query_partition'] == [b+1 for b in case['budgets']]+case['scratch_Q']
                    and t['dense_AR_qualified'] is t['cross_M_equality_qualified'] is False,
                    'exact independent same-M/S geometry and local scope')
                raw_mc2(t['MC2_candidate'],rank)
                raw_mc2(t['MC2_independent_eager'],rank)
                commit_proof(q['target_commit_oracle'])
                commit_proof(q['plain_restored_commit_oracle'])
            if q['changed_input_variant'] is False:
                if q['method'] in qualified:
                    require(qualified[q['method']] == observed_signature(q), 'F/T/F restored original path')
                qualified[q['method']] = observed_signature(q)
        for row in [*[r for trace in report['fixed_point'] for r in trace],*report['cost']]:
            protocol_proof(row,wanted_protocol,row['case'])
            require(observed_signature(row) == qualified[row['method']], 'plain replay exactly follows own qualified path')
            true_fields(row, ('original_root_diagnostics_retained','permanent_tail_observer_retained',
                'local_sequential_round_plus_nextproposal','actual_observed_verdict_only',
                'original_root_diagnostics_retained_in_cost','original_tail_observer_retained_in_cost'),
                'unchanged timed observers/actual verdict')
            require(row['extra_scoped_first_observer'] is False
                and row['production_diagFalse_cost_qualified'] is False, 'no scoped observer/cold plain path claim')
            require(len(row['g']) == len(row['emitted']) == len(row['accepted']) == 16,
                    'all16 actual observed verdict rows')
            require(all(type(g) is int and g == len([t for t in out if t >= 0])
                for g,out in zip(row['g'],row['emitted'])), 'g exactly counts actual delivered tokens')
            require(row['measured_success_MC2_evidence_flushed_after_elapsed'] == (0 if rank == 0 else 1),
                    'successful MC2 raw evidence outside measured elapsed')
    witness_paths = sorted(directory.glob('rank0-native-witness-*.json'))
    require([read(p) for p in witness_paths] == witnessed, 'all original partial/final raw witness evidence retained')
    raw_evidence.update({str(p):sha(p) for p in witness_paths})
    mc2_records = []
    for rank in (1,2,3):
        mc2_paths = sorted(directory.glob(f'rank{rank}-MC2-*.json'))
        # 9 qualification calls each candidate+independent+plain, then 6+15 replay calls.
        require(len(mc2_paths) == 48, 'all48 actual MC2 records per target rank')
        serials = [int(p.name.split('-')[2]) for p in mc2_paths]
        require(serials == list(range(1,49)), 'continuous actual MC2 evidence serials')
        for p in mc2_paths:
            observation = raw_mc2(read(p),rank)
            mc2_records.append(dict(path=str(p),sha256=sha(p),rank=rank,**observation))
            raw_evidence[str(p)] = sha(p)
        selected = [r for r in mc2_records if r['rank'] == rank]
        require(sum(r['mode'] == 'independent_eager' for r in selected) == 9,
                'nine fresh same-M eager targets per rank')
    require(receipt_path.read_bytes() == receipt_bytes, 'terminal receipt unchanged while reading')
    return receipt,reports,completed,frozen_audit,raw_evidence,mc2_records,wanted_protocol,runtime_outputs


def summarize(reports):
    families = []
    for name in NAMES:
        rows = [[r for r in report['cost'] if r['method'] == name] for report in reports]
        require(all([r['sample'] for r in rr] == list(range(5)) for rr in rows), 'five paired sample IDs')
        sample = rows[0][0]
        require(all(observed_signature(r) == observed_signature(sample) for r in rows[0]),
                'same-method repeated actual g/verdict')
        families.append(dict(method=name,case=sample['case'],
            rank0_local_wall_cost=stats(r['sequential_local_round_plus_nextproposal_wall_ms'] for r in rows[0]),
            synchronized_whole_wall_rank_max_per_call=stats(max(rows[rank][s]['sequential_local_round_plus_nextproposal_wall_ms']
                for rank in range(4)) for s in range(5)),
            rank0_phase_costs_ms={p:stats(r['local_phase_wall_ms'][p] for r in rows[0]) for p in PHASES},
            target_rank_max_per_call_ms=stats(max(rows[rank][s]['local_phase_wall_ms']['target_whole_forward_ms']
                for rank in (1,2,3)) for s in range(5)),
            per_rank_cost=[dict(rank=rank,whole=stats(r['sequential_local_round_plus_nextproposal_wall_ms'] for r in rr),
                phases={p:stats(r['local_phase_wall_ms'][p] for r in rr) for p in PHASES})
                for rank,rr in enumerate(rows)],
            g_per_owner=sample['g'],sum_actual_delivered_tokens=sum(sample['g']),
            sum_accepted_nodes=sum(t >= 0 for rr in sample['accepted'] for t in rr),
            emitted_per_owner=sample['emitted'],accepted_packed_indices_per_owner=sample['accepted'],
            selected_original_nodes=sample['selected_original_nodes'],
            next_missing_suffix_lengths=sample['next_missing_suffix_lengths'],
            next_draft_query_count=sample['next_draft_query_count'],next_draft_counters=sample['next_draft_counters'],
            pooling_into_gfit_qualified=False))
    comparisons = []
    for a,b in itertools.combinations(families,2):
        paired = []
        for sample in range(5):
            aa = next(r for r in reports[0]['cost'] if r['method'] == a['method'] and r['sample'] == sample)
            bb = next(r for r in reports[0]['cost'] if r['method'] == b['method'] and r['sample'] == sample)
            paired.append(dict(sample=sample,
                first_minus_second_whole_ms=aa['sequential_local_round_plus_nextproposal_wall_ms']-bb['sequential_local_round_plus_nextproposal_wall_ms'],
                first_minus_second_target_ms=aa['local_phase_wall_ms']['target_whole_forward_ms']-bb['local_phase_wall_ms']['target_whole_forward_ms']))
        comparisons.append(dict(first=a['method'],second=b['method'],rank0_paired_differences=paired,
            common_source_fixed_repeats_not_independent_trials=True,
            delivered_g_not_normalized_or_pooled=True,production_speedup_qualified=False))
    signatures = {f['method']:next(r for r in reports[0]['cost'] if r['method'] == f['method']) for f in families}
    cross = [signature_differences(a,b,signatures[a],signatures[b]) for a,b in itertools.combinations(NAMES,2)]
    for report in reports:
        require(report['cross_layout_actual_emission_differences'] == cross, 'recomputed cross-M difference disclosure')
    return dict(schema='v91-actual-causal80-cost-and-delivered-tokens-v1',
        result_scope='Same16 source-fixed sequential local round plus actual nextproposal only',
        synchronized_cost_calls=15,cases=3,samples_per_case=5,rank_observations_per_call=4,
        rank_observations_are_not_independent_samples=True,
        same16source_histories_repeated_not_independent_samples=True,
        source_prefill_and_independent_references_outside_elapsed=True,
        original_root_retention_True_and_permanent_tail_observer_in_elapsed=True,
        phase_medians_are_not_additive_and_do_not_partition_whole_elapsed=True,
        rank_max_per_call_is_a_synchronous_wall_observation_not_isolated_compute=True,
        family_summary=families,paired_comparisons=comparisons,cross_M_actual_output_differences=cross,
        independent_own_AR_qualified=False,production_diagFalse_cost_qualified=False,
        production_overlap_qualified=False,formal_Goodput=False,cross_M_pooling_into_gfit_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    args = parser.parse_args()
    try:
        receipt,reports,completed,original,evidence,mc2,protocol_hashes,runtime_outputs = audit(args.tag)
        summary = summarize(reports)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        summary_path = OUT/'V91_CAUSAL80_COST_AND_DELIVERED_TOKENS.json'
        audit_path = OUT/'ROOT_V91_ACTUAL_CAUSAL80_COST_AUDIT.json'
        audit_result = dict(schema='v91-actual-causal80-external-root-audit-v1',audited_UTC=now,passed=True,
            receipt=str(WORK/('cost-receipt-'+args.tag+'.json')),receipt_status='complete',
            original_frozen_report_audit_recomputed=original,source_manifest_sha256=FREEZE_SHA,
            local_frozen_files=len(receipt['input_hashes']),external_frozen_files=len(receipt['external_input_hashes']),
            source_archive=str(receipt['source_snapshot']),source_archive_sha256=receipt['source_snapshot_sha256'],
            qualification_records_per_rank=9,F_T_F_original_changed_restored_per_case=True,
            complete_zero_discovery_traces_per_rank=2,sealed_before_cost=True,
            synchronized_cost_calls=15,cases=3,samples_per_case=5,rank_observations=60,
            rank_observations_are_not_independent_samples=True,
            exact_zero_native_fields_per_rank_per_trace_or_cost_call=12,
            expected_zero_native_values=4*(6+15)*12,
            draft_all28_and_target_all64_query_KV_commit_and_nextproposal_checked=True,
            original_protocol_decode_sha256=protocol_hashes[0],canonical_protocol_sha256=protocol_hashes[1],
            MC2_raw_count=len(mc2),MC2_records=mc2,evidence_paths_sha256=evidence,
            elapsed_seconds=receipt['elapsed_seconds'],private_bytes=receipt['private_bytes'],
            pre_and_post_verifier_exit0=True,owned_remaining=[],cards_idle_before_after=True,
            source_environment_closure_unchanged=True,
            root_actual_pre_NPU_audit_sha256=PRE_AUDIT_SHA,
            postrun_launch_readiness_collect_reexecuted=False,
            known_runtime_outputs=runtime_outputs,
            environment_evidence_scope='Unchanged pinned runtime shell/source closure and original pre/post verifier; no invented process environment snapshot',
            resource_evidence_limit='Original guards execute around methods. Per-cost peak/free samples are not persisted; no measured cost-peak claim.',
            cost_summary=str(summary_path),independent_own_AR_qualified=False,
            production_overlap_qualified=False,formal_Goodput=False,cross_M_pooling_into_gfit_qualified=False,
            analyzer_sha256=sha(__file__),torch_imported=False,NPU_launched=False)
        # No output artifact exists until terminal and deep gates have all passed.
        summary_bytes = (json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
        audit_result['cost_summary_sha256'] = hashlib.sha256(summary_bytes).hexdigest()
        audit_bytes = (json.dumps(audit_result,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
        summary_path.write_bytes(summary_bytes)
        audit_path.write_bytes(audit_bytes)
        print(json.dumps(dict(passed=True,audit=str(audit_path),summary=str(summary_path),
            synchronized_calls=15,independent_rank_samples=False),ensure_ascii=False))
        return 0
    except (OSError,KeyError,ValueError,RuntimeError,AssertionError,StopIteration,tarfile.TarError) as error:
        print(json.dumps(dict(passed=False,result_files_written=False,error=str(error)),ensure_ascii=False),file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
