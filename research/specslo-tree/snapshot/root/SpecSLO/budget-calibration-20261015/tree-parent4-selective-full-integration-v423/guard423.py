"""Single-owner full423; original227 plus681 bucket cases and owned cleanup."""
import argparse, ast, copy, datetime, fcntl, gzip, hashlib, importlib.util, json, os, re, resource, signal, subprocess, sys, tarfile, time, traceback
from pathlib import Path
from check423 import P as HERE, B, DONOR, OLD, OLD_SHA, PARENT_FREEZE, require, ready, read, sha
sys.path[1:1] = [str(OLD), str(DONOR)]
sys.path.extend([str(B/'tree-static-q5-M120-full463-own-AR-v256'),str(B/'tree-static-q5-M120-node-math-v255'),str(B/'tree-static-q5-M128-full463-own-AR-v251'),str(B/'tree-static-q5-M128-projection-host-v249'),str(B/'tree-static-q5-immutable-index-cache-host-v247'),str(B/'tree-static-q5-long-prefix-origins-node-math-v246'),str(B/'tree-static-q5-projection-host-v243'),str(B/'tree-canonical-contiguous-full463-own-AR-v235'),str(B/'tree-contiguous-branch-tail-final-metadata-math-v234')])
from guard218 import donor_guard

LOCK = B/'.ROOT_NPU_EXECUTION.lock'
MAX_SECONDS = 18000
MAX_TAG_BYTES = 256*1024**2
MAX_AGGREGATE_BYTES = 3584*1024**2
EXPECTED_FREEZE = None
EXPECTED_ARCHIVE = None
ACTION = 'causal80-N9-to15-allB4-empty-active32-full463'


def prepared():
    frozen = ready(EXPECTED_FREEZE)
    sources = dict(frozen['files'])
    sources[str(HERE/'FREEZE423.json')] = EXPECTED_FREEZE
    archive = HERE/'SOURCE423.tar.gz'
    require(sha(archive) == EXPECTED_ARCHIVE, 'explicit256 archive pin')
    with tarfile.open(archive, 'r:gz') as saved:
        members = saved.getmembers()
        require(len(members) == len(sources) and {m.name for m in members} == {p.lstrip('/') for p in sources}, 'exact256 archive inventory')
        for member in members:
            require(member.isfile() and hashlib.sha256(saved.extractfile(member).read()).hexdigest() == sources['/'+member.name], 'exact archived256 source '+member.name)
    original = read(OLD/'FREEZE218.json')
    require(all(frozen['files'].get(p) == h for p,h in original['files'].items()), 'all original218 sources/evidence retained')
    return frozen, sources, dict(own_freeze_sha256=EXPECTED_FREEZE, own_archive_sha256=EXPECTED_ARCHIVE,
        original218_freeze_sha256=OLD_SHA, device_qualification_inherited=False,
        production_delta='original startup/components plus227+681 and fresh same-engine75 parent4 lifecycle cases; explicit selective N24/E4/B4 dispatcher; global64; original463 arrivals/SLO and all3 own-AR')


def storage(tag=None):
    original = donor_guard()
    actual = 0 if tag is None else original.bytes_under(DONOR/'actual'/tag)
    total = original.bytes_under(DONOR)+original.bytes_under(HERE)
    require(actual <= MAX_TAG_BYTES and total <= MAX_AGGREGATE_BYTES, 'private256MiB per-tag /3584MiB aggregate bounded files')
    return dict(actual_tag_bytes=actual, aggregate_private_bytes=total)


def launch_argv(tag):
    return ['/usr/bin/env', '-i', 'HOME=/root', 'USER=root', 'LOGNAME=root', 'SHELL=/bin/bash', 'LANG=C.UTF-8',
        'PATH=/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin',
        'SPECSLO_V257_ROOT_FREEZE_SHA='+PARENT_FREEZE, 'SPECSLO_V376_ROOT_FREEZE_SHA=ae723ddfddeb6d47f04605a3842de0c3690196b0addf39fa17c5bd1ca248765c', '/bin/bash', str(HERE/'launch423.sh'), '--action', ACTION, '--tag', tag]


def actual_audit(directory):
    from audit423 import audit
    actual = audit(directory, (64,))
    completed = read(Path(directory)/'completed.json')
    exited = read(Path(directory)/'worker-exit.json')
    require(completed['action'] == ACTION and completed['audit'] == actual and actual['passed'] is True
        and completed['replies'] == [['calibrated', r, dict(windows=[64], full463=True)] for r in range(4)]
        and exited == dict(closed=True, all_exited=True), 'complete full workers and original owned cleanup')
    for rank in range(4):
        report = read(Path(directory)/f'rank{rank}-local-diagnostic.json')
        trace = json.loads(gzip.decompress((Path(directory)/'windows/window64/full463'/f'rank{rank}-full463-actual-trace.json.gz').read_bytes()))
        from cost218 import audit_snapshot
        audit_snapshot(report['ROOT218_cached_parent_dispatch'], trace['source_generation'])
        from slot_preflight218 import audit_snapshot
        audit_snapshot(report['ROOT218_slot_preflight'], rank, trace)
        require(report['postfull_independent_FIA_AR'] == read(Path(directory)/'postfull-independent-FIA-AR'/f'rank{rank}-summary.json'), 'actual independent reference report retained')
    require(actual['whole_workload_AR_qualified'] is True, 'all463 complete outputs must equal independent own-AR')
    return actual, 'passed_current_canonical_full_AR', dict(raw_native_zero_subtractions=actual['native_zero_subtractions'],
        hidden_KV_recomputed_by_HOST=False, whole_workload_AR_qualified=True)


def execution_namespace():
    original = donor_guard()
    namespace = dict(vars(original), P=DONOR, G=HERE, ACTION=ACTION, prepared=prepared,
        storage=storage, launch_argv=launch_argv, actual_audit=actual_audit, MAX_SECONDS=MAX_SECONDS,
        MAX_TAG_BYTES=MAX_TAG_BYTES, MAX_AGGREGATE_BYTES=MAX_AGGREGATE_BYTES,
        SCOPE=dict(original201_44_114_and_all_original_components_required=True,
                   original_global_ENV_TARGET0_unchanged=True, actual_LIVE_target_SLO_first=True,
                   fresh_same_instance_canonical_math_cases=227, fresh_same_instance_uniform_KV_cases=681,
                   original_best257_optimized_draft_unchanged=False,
                   original_RootSeed_and_legacy_Fast_fallback_retained=True,
                   current_same_engine_parent4_lifecycle_cases=75,
                   normal_work_admission_policy_limit=24,
                   original_gain_model_unchanged=True,
                   original_gain_model_and_typed_role_limits_unchanged=False,
                   new_exact_paid_work=dict(normal24_B4=96,future4_B4=16,total=112),
                   active_request_limit=64, previous_active_request_limit=64, historical_best_tree_active_request_limit=48, target_attention_segment_limit=24, full463=True,
                   actual_new_canonical_graph_keys=True, effective_requests=24, effective_candidates=96,
                   configured_requests=32, configured_candidates=96,
                   original_models_arrivals_SLO_and_sampler=True, postfull_own_AR_required=True,
                   formal_Goodput=False, adopted=False))
    source = (B/'budget-diagnostics/v154-root-execution/supervise_v154.py').read_text()
    replacements = (
        ('V154.root-owned-local-component-supervisor.v1', 'ROOT.V423.best257-uniform-KV-full-own-AR-supervisor.v1'),
        ('V154 supervisor received', 'ROOT423 supervisor received'),
        ('numeric_source_obligations_only=True', 'numeric_source_obligations_only=False'),
        ('one natural-positive R1 discovery, conditional new-union/child/next; no two complete zeros or full qualification',
         'original startup/components plus227 original FULL target cases and681 uniform-KV target cases; static q5/M120/KV2560 and contiguous branch KV, max24 primary/side groups; active64 with page reservation; warm/twozero/seal/full463/independent own-AR'),
        ('bounded1800second', 'bounded18000second'), ('within1800seconds', 'within18000seconds'),
        ("'actual_component_audit'", "'actual_full_audit'"), ("'actual_local_coverage_status'", "'actual_full_status'"))
    for before, after in replacements:
        require(source.count(before) == 1, 'original supervisor metadata anchor '+before)
        source = source.replace(before, after)
    names = {'verifier_report', 'child_limits', 'execute_locked'}
    nodes = [copy.deepcopy(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == names, 'all original verifier, limits and owned cleanup bodies')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), __file__+'::original-supervision', 'exec'), namespace)
    return namespace


def main():
    global EXPECTED_FREEZE, EXPECTED_ARCHIVE
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--expected-freeze-sha256', required=True)
    parser.add_argument('--expected-archive-sha256', required=True)
    args = parser.parse_args()
    EXPECTED_FREEZE = args.expected_freeze_sha256
    EXPECTED_ARCHIVE = args.expected_archive_sha256
    require(all(re.fullmatch('[0-9a-f]{64}', s) is not None for s in (EXPECTED_FREEZE, EXPECTED_ARCHIVE))
            and re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', args.tag) is not None, 'explicit pins and fresh safe tag')
    prepared()
    namespace = execution_namespace()
    lock = LOCK.open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
