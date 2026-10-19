"""Single-owner bounded target-model component with original environment checks."""
import argparse
import ast
import copy
import datetime
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import tarfile
import time
import traceback
HERE = Path(__file__).resolve().parent
B = HERE.parent
PARENT = B / 'tree-FIA-Q4-group-and-query-isolation-v335'
sys.path.insert(0, str(PARENT))
import guard335 as parent
from geometry336 import cases, cost_arms, LAYOUTS
LOCK = B / '.ROOT_NPU_EXECUTION.lock'
MAX_SECONDS = 1800
MAX_TAG_BYTES = 96 * 1024 ** 2
MAX_AGGREGATE_BYTES = 256 * 1024 ** 2
ACTION = 'linear-Q4-Q5-target-shape-node-cost-component'
EXPECTED_FREEZE = None
EXPECTED_ARCHIVE = None
require = parent.require
sha = parent.sha

def prepared():
    require(sha(HERE / 'FREEZE336.json') == EXPECTED_FREEZE, 'exact additive336 source pin')
    own = json.loads((HERE / 'FREEZE336.json').read_text())
    for path, expected in own['files'].items():
        require(sha(path) == expected, 'immutable336 input ' + path)
    parent_pins = json.loads((PARENT / 'SOURCE335_RECEIPT.json').read_text())
    require(own['parent_freeze_sha256'] == parent_pins['freeze_sha256'] and own['parent_archive_sha256'] == parent_pins['archive_sha256'], 'explicit parent335 source-only pins')
    parent.EXPECTED_FREEZE = parent_pins['freeze_sha256']
    parent.EXPECTED_ARCHIVE = parent_pins['archive_sha256']
    _, sources, parent_ready = parent.prepared()
    own_sources = dict(own['files'])
    own_sources[str(HERE / 'FREEZE336.json')] = EXPECTED_FREEZE
    require(sha(HERE / 'SOURCE336.tar.gz') == EXPECTED_ARCHIVE, 'exact additive336 archive')
    with tarfile.open(HERE / 'SOURCE336.tar.gz', 'r:gz') as saved:
        members = saved.getmembers()
        require({m.name for m in members} == {p.lstrip('/') for p in own_sources} and len(members) == len(own_sources), 'exact additive archive inventory')
        for member in members:
            require(member.isfile() and hashlib.sha256(saved.extractfile(member).read()).hexdigest() == own_sources['/' + member.name], 'exact additive archive bytes')
    require(not set(sources) & set(own_sources), 'disjoint immutable additive files')
    sources.update(own_sources)
    return (own, sources, dict(own_freeze_sha256=EXPECTED_FREEZE, own_archive_sha256=EXPECTED_ARCHIVE, own_archive_members=len(own_sources), parent_source_archive_verified_in_place=parent_ready, source_members=len(sources), model_qualification_inherited=False))

def storage(tag=None):
    helper = parent.parent.donor_guard()
    actual = 0 if tag is None else helper.bytes_under(HERE / 'actual' / tag)
    total = helper.bytes_under(HERE)
    require(actual <= MAX_TAG_BYTES and total <= MAX_AGGREGATE_BYTES, 'private target diagnostic data bound')
    return dict(actual_tag_bytes=actual, aggregate_private_bytes=total)

def launch_argv(tag):
    return ['/usr/bin/env', '-i', 'HOME=/root', 'USER=root', 'LOGNAME=root', 'SHELL=/bin/bash', 'LANG=C.UTF-8', 'PATH=/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin', 'SPECSLO_V336_ROOT_FREEZE_SHA=' + EXPECTED_FREEZE, 'SPECSLO_V218_ROOT_FREEZE_SHA=fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260', '/bin/bash', str(B / 'serial-v1/runtime.sh'), '/bin/bash', str(HERE / 'startup336.sh'), tag]

def actual_audit(directory):
    directory = Path(directory)
    reports = [json.loads((directory / f'rank{rank}-numeric.json').read_text()) for rank in range(4)]
    expected = cases()
    require(len(expected) == 36, 'explicit finite36 linear node cases')
    head_sets = []
    cost_rows = []
    keys = ('aclgraph_captures', 'aclgraph_capture_attempts', 'aclgraph_runtime_validation_replays', 'aclgraph_failed_captures', 'aclgraph_capacity_fallbacks', 'aclgraph_shape_fallbacks')
    arms = cost_arms()
    names = [a['name'] for a in arms]
    for rank, report in enumerate(reports):
        require(report['rank'] == rank and report['case_count'] == 36 and (len(report['cases']) == 36), 'all four actual current reports')
        require(report['performance'] is report['adopted'] is report['real_draft_generation'] is False and report['whole_workload_AR_qualified'] is False, 'component only')
        if rank == 0:
            require(all((row is None for row in report['cases'])) and report['bank'] is None, 'draft role makes no target numeric claim')
        else:
            require(report['model_dtype'] == 'torch.bfloat16' and report['all_prefixes_exact_after_cost'] is True, 'BF16 and protected own prefixes')
            bank = report['bank']
            require(bank['independent_neural_prefills'] == 96 and bank['all_triplet_roots_equal'] is bank['all_triplet_prefix_KV_equal'] is True, 'fresh96 independent neural prefix bank')
            activation = report['activation']
            require(activation['original_model_forward_unchanged'] is True and activation['Native_flags_assigned'] is False and (activation['deferred_read_done'] is False), 'original actual-row MC2 model, no assigned qualification flags')
            require(len(report['MC2_routes']['routes']) == 128, 'all128 actual TP3 route modules')
            before_metrics = report['original_engine_metrics_before']
            after_metrics = report['original_engine_metrics_after']
            for suffix in ('fallback', 'exception'):
                key = 'mc2_dispatch_' + suffix
                require(after_metrics[key] == before_metrics[key], 'no actual MC2 fallback/exception')
            for suffix in ('native_epilogue_chained_success', 'native_epilogue_chain_flush'):
                key = 'mc2_dispatch_' + suffix
                require(after_metrics[key] > before_metrics[key], 'actual original flushed MC2 chain dispatched')
            heads = []
            for case, row in zip(expected, report['cases'], strict=True):
                require(row['case'] == case and [v['future'] for v in row['comparisons']] == [0, 17], 'exact real geometry and both independent masked-future canaries')
                for comparison in row['comparisons']:
                    require(all((comparison[k] is True for k in ('exact_greedy', 'hidden_bitwise_equal', 'all128_KV_bitwise_equal', 'graph_eager_KV_bitwise_equal'))) and comparison['candidate_KV_copied'] is False, 'all original and strict real-node checks')
                    require(comparison['candidate_tokens'] == comparison['reference_tokens'] and len(comparison['candidate_tokens']) == case['query_count'], 'all independent node heads')
                    require([frame['depth'] for frame in comparison['frames']] == list(range(case['width']))
                            and all(frame['real_nodes'] == case['N'] and frame['physical_queries'] == case['M']
                                    and frame['future'] == comparison['future']
                                    and frame['candidate_KV_copied'] is False
                                    and frame['reference_eager_only'] is True for frame in comparison['frames']),
                            'complete independent own-history node frames')
                require(row['comparisons'][0]['reference_tokens'] == row['comparisons'][1]['reference_tokens'], 'masked future cannot affect real predictions')
                heads.append(row['comparisons'][0]['candidate_tokens'])
            head_sets.append(heads)
            require(len(report['cold_native_calls']) == 36, 'one actual Native target invocation per finite case')
            for call in report['cold_native_calls']:
                width, groups = LAYOUTS[call['layout']]
                require(call['M'] == width * groups and call['groups'] == groups and (call['width'] == width) and (call['mask_shape'] == [groups, 1, 10, 4096]) and (call['actual_seq_lengths_q'] == list(range(width, width * groups + 1, width))) and (call['sequence_lens'] == [2560] * groups) and (call['used_aclgraph'] is True) and (call['fallback_reason'] is None), 'actual Native graph metadata and execution')
        costs = report['paired_target_cost']
        require(costs['rounds'] == len(costs['pairs']) == 20 and costs['no_new_graph_work'] is True and all((costs['graph_counters_before'][k] == costs['graph_counters_after'][k] for k in keys)), 'untimed captures excluded from rotated costs')
        require(costs['arms'] == list(arms), 'all four exact cost arms')
        samples = {name: [] for name in names}
        for index, pair in enumerate(costs['pairs']):
            require(pair['order'] == names[index % 4:] + names[:index % 4] and set(pair['milliseconds']) == set(names), 'actual balanced rotation')
            for name in names:
                value = pair['milliseconds'][name]
                if rank == 0:
                    require(value is None, 'draft has no invented target elapsed time')
                else:
                    require(type(value) in (int, float) and math.isfinite(value) and (value > 0), 'positive target elapsed')
                    samples[name].append(value)
        if rank:
            import statistics
            cost_rows.append(dict(rank=rank, median_ms={name: statistics.median(values) for name, values in samples.items()}))
    require(head_sets[0] == head_sets[1] == head_sets[2], 'all3 actual target greedy heads agree')
    completed = json.loads((directory / 'completed.json').read_text())
    exited = json.loads((directory / 'worker-exit.json').read_text())
    require(completed['replies'] == [['calibrated', rank, dict(numeric_cases=36, performance=False)] for rank in range(4)] and exited == dict(closed=True, all_exited=True), 'complete original worker lifecycle')
    return (dict(status='complete_current_target_component', cases_each_target=36, canaries=2, costs=cost_rows, whole_workload_AR_qualified=False, real_draft_generation=False, actual_dynamic_B_or_tree_branch_qualified=False, formal_Goodput=False, adopted=False), 'passed_finite_linear_model_math_and_cost', dict(real_node_heads_cross_rank=True, paired_costs_recomputed=True))

def execution_namespace():
    original = parent.parent.donor_guard()
    namespace = dict(vars(original), P=HERE, G=HERE, ACTION=ACTION, prepared=prepared, storage=storage, launch_argv=launch_argv, actual_audit=actual_audit, MAX_SECONDS=MAX_SECONDS, MAX_TAG_BYTES=MAX_TAG_BYTES, MAX_AGGREGATE_BYTES=MAX_AGGREGATE_BYTES, SCOPE=dict(models_loaded=True, full463=False, formal_Goodput=False, adopted=False, real_target_roles=3, finite_cases=36, independent_prefixes=96, real_draft_generation=False, tree_branch_qualified=False))
    source = (B / 'budget-diagnostics/v154-root-execution/supervise_v154.py').read_text()
    changes = (('V154.root-owned-local-component-supervisor.v1', 'ROOT.V336.linear-target-shape-supervisor.v1'), ('V154 supervisor received', 'ROOT336 supervisor received'), ('one natural-positive R1 discovery, conditional new-union/child/next; no two complete zeros or full qualification', 'fresh target Q4/Q5 node math, two masked-future canaries, paired target costs; no real draft or service workload'), ('bounded1800second', 'bounded1800second'), ('within1800seconds', 'within1800seconds'), ("'independent_native_discovery_arithmetic'", "'independent_component_arithmetic'"))
    for before, after in changes:
        require(source.count(before) == 1, 'exact original metadata anchor ' + before)
        source = source.replace(before, after)
    names = {'verifier_report', 'child_limits', 'execute_locked'}
    nodes = [copy.deepcopy(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == names, 'original owned supervision bodies retained')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), __file__ + '::original-owned-supervision', 'exec'), namespace)
    return namespace

def main():
    global EXPECTED_FREEZE, EXPECTED_ARCHIVE
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--expected-freeze-sha256', required=True)
    parser.add_argument('--expected-archive-sha256', required=True)
    args = parser.parse_args()
    EXPECTED_FREEZE, EXPECTED_ARCHIVE = (args.expected_freeze_sha256, args.expected_archive_sha256)
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', args.tag) is not None and all((re.fullmatch('[0-9a-f]{64}', s) for s in (EXPECTED_FREEZE, EXPECTED_ARCHIVE))), 'explicit safe tag/pins')
    prepared()
    namespace = execution_namespace()
    lock = LOCK.open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
if __name__ == '__main__':
    main()
