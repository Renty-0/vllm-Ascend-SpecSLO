"""One terminal-only stdlib audit; no model imports, queries or replay."""
import argparse
import ast
import hashlib
import importlib.abc
import json
from pathlib import Path
import re
import sys
import tarfile

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B/'tree-split192-live-v132'
D = B/'budget-diagnostics'
EXPECTED_RECEIPT_SHA = 'f14deee61fbd932bf6b5970bcc3364132e704c1b507a4bbc56be933a39df92ab'
ATTEMPTS = []
FORBIDDEN = {'torch', 'torch_npu', 'vllm', 'vllm_ascend'}


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            ATTEMPTS.append(fullname)
            raise ImportError('HOST-only failed-run audit rejects '+fullname)


sys.meta_path.insert(0, Guard())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def load(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify_map(values):
    bad = [name for name, wanted in values.items() if not Path(name).is_file() or sha(name) != wanted]
    require(not bad, 'Source/readonly input changed: '+repr(bad[:5]))
    return len(values)


def archive(path, expected, *, member_prefix=''):
    seen = set()
    with tarfile.open(path, 'r:gz') as stream:
        for member in stream:
            require(member.name.startswith(member_prefix), 'Source archive prefix differs')
            name = '/'+member.name[len(member_prefix):]
            require(member.isfile() and name in expected and name not in seen,
                    'Unexpected/duplicate/non-file source archive member: '+member.name)
            require(hashlib.sha256(stream.extractfile(member).read()).hexdigest() == expected[name],
                    'Source archive member bytes differ: '+name)
            seen.add(name)
    require(seen == set(expected), 'Source archive closure incomplete')
    return dict(**record(path), members_all_bytes_checked=len(seen))


def fusion(path):
    require(0 < path.stat().st_size <= 64*1024**2, 'Fusion diagnostic size')
    pairs = json.loads(path.read_text(), object_pairs_hook=lambda rows: rows)
    require(type(pairs) is list and bool(pairs), 'Fusion root')
    for name, graph in pairs:
        require(re.fullmatch(r'session_and_graph_id_\d+_\d+', name) and type(graph) is list
                and len(graph) == 1 and graph[0][0] == 'graph_fusion', 'Fusion graph')
        passes = graph[0][1]
        require(type(passes) is list and bool(passes) and len({n for n, _ in passes}) == len(passes), 'Fusion passes')
        for name, values in passes:
            require(name in {'TransdataCastFusionPass', 'TransdataFz2FzgFusionPass', 'TransdataFzg2FzFusionPass'}
                    and type(values) is list and len(values) == 2
                    and {n for n, _ in values} == {'effect_times', 'match_times'}
                    and all((type(v) is int and v >= 0) or (type(v) is str and re.fullmatch(r'\d+', v)) for _, v in values),
                    'Fusion pass counters')
    return dict(**record(path), graph_records=len(pairs), retained_not_source=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.tag), 'Invalid tag')
    rp = P/('local-receipt-'+args.tag+'.json')
    r = load(rp)
    require(sha(rp) == EXPECTED_RECEIPT_SHA, 'ROOT terminal receipt identity')
    require(r['status'] == 'failed' and r['child']['exit_code'] == 1 and bool(r['errors']), 'Original failure must remain failure')
    require(r['owned_remaining'] == [] and r['devices_after']['all_idle'] is True
            and r['pre_verifier']['exit_code'] == r['post_verifier']['exit_code'] == 0
            and r['experimental_sources_unchanged'] is r['external_sources_unchanged'] is True,
            'Original terminal cleanup/source/verifier evidence')
    require('report_paths' not in r and 'local_execution_complete' not in r, 'Do not infer successful worker reports')
    cpu_path, frozen_path = P/'CPU_RESULT_V132.json', P/'PREPARED_SOURCE_HASHES_V132.json'
    cpu, frozen = load(cpu_path), load(frozen_path)
    require(cpu['status'] == 'passed' and cpu['tests_run'] == 23 and not cpu['errors'] and not cpu['failures']
            and cpu['NPU_initialized'] is False and frozen['CPU_sha256'] == sha(cpu_path), 'Original HOST/freeze evidence')
    require(cpu['input_hashes'] == cpu['input_hashes_after'] == frozen['files']
            and cpu['external_input_hashes'] == cpu['external_input_hashes_after'] == frozen['external_files'], 'CPU before/after source closure')
    source_counts = dict(local=verify_map(frozen['files']), external=verify_map(frozen['external_files']),
                         runtime_archived=verify_map(r['source_hashes']), receipt_external=verify_map(r['external_input_hashes']))
    require(r['readiness']['identity'] == dict(CPU_sha256=sha(cpu_path), freeze_sha256=sha(frozen_path)), 'Actual retained readiness identity')
    require(r['source_hashes'] == {**frozen['files'], **frozen['external_files'],
            str(cpu_path):sha(cpu_path), str(frozen_path):sha(frozen_path)}, 'Runtime archive exact declared closure')
    require(sha(r['source_snapshot']) == r['source_snapshot_sha256'], 'Runtime archive SHA')
    runtime_archive = archive(r['source_snapshot'], r['source_hashes'])
    pre_path = D/'ROOT_V132_PRE_NPU_SOURCE_AND_HOST_AUDIT.json'
    pre = load(pre_path)
    require(pre['status'] == 'passed' and pre['CPU_sha256'] == sha(cpu_path)
            and pre['freeze_sha256'] == sha(frozen_path) and pre['readiness_exit_code'] == 0, 'ROOT original preaudit')
    require(sha(pre['archive']) == pre['archive_sha256'], 'Prepared archive SHA')
    prepared_archive = archive(pre['archive'], r['source_hashes'], member_prefix='inputs/')
    shells = re.findall(r'^source (/[^ \n]+)', (B/'serial-v1/runtime.sh').read_text(), re.MULTILINE)
    require([row['path'] for row in r['actual_sourced_runtime_shells_before']] == shells
            and r['actual_sourced_runtime_shells_before'] == r['actual_sourced_runtime_shells_after']
            and all(sha(row['path']) == row['sha256'] for row in r['actual_sourced_runtime_shells_after']), 'Actual two sourced shells')
    require(r['runtime_aliases_before'] == r['runtime_aliases_after'], 'Runtime aliases changed')
    for row in r['runtime_aliases_after']:
        path = Path(row['path'])
        require(path.is_symlink() and str(path.resolve()) == row['resolved'] and path.readlink().as_posix() == row['link']
                and path.exists() is row['exists'], 'Actual runtime alias differs')
    out = P/'local-output'/args.tag
    require({path.name for path in out.iterdir()} == {*(f'rank{rank}-last-error.json' for rank in range(4)), 'worker-exit.json'},
            'Unexpected qualification/trace output domain')
    errors = []
    source = (P/'candidate_runtime_v132.py').read_text()
    tree = ast.parse(source)
    failure_call = next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n) == "__import__('dataclasses').asdict(engine.config)")
    qualification_call = next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func) == 'qualify_structures')
    require(failure_call.lineno == 88 and failure_call.lineno < qualification_call.lineno, 'Failure/qualification source order')
    for rank in range(4):
        path = out/f'rank{rank}-last-error.json'
        value = load(path)
        require(value['rank'] == rank and value['action'] == r['action'] and value['formal_Goodput'] is value['online_qualified'] is False
                and 'candidate_runtime_v132.py\", line 88, in worker' in value['error']
                and value['error'].endswith("TypeError: cannot pickle 'mappingproxy' object\n"), 'Actual identical first failure')
        errors.append(dict(**record(path), rank=rank, error=value['error']))
    exited = load(out/'worker-exit.json')
    require(exited == dict(closed=True,all_exited=True), 'Actual worker exit')
    startup_dir = Path(r['startup_progress']['actual_directory'])
    require(startup_dir == P/'startup-progress-by-tag'/args.tag and (P/'startup-progress').resolve() == startup_dir, 'Actual tag-private startup')
    startup = []
    for rank in range(4):
        mc2_path = startup_dir/f'MC2-preflight-rank{rank}.json'; mc2 = load(mc2_path)
        require(mc2['status'] == 'passed' and mc2['all4rank_contract_passed'] is True and mc2['local_passed'] is True
                and mc2['failed_checks'] == [] and len(mc2['planned_full_chain_rows']) == 22
                and mc2['native_qualification_flags_modified'] is False
                and sha(mc2['profile_path']) == mc2['profile_disk_sha256'], 'Actual startup MC2 receipt')
        paths = [mc2_path]
        if rank == 0:
            dp = startup_dir/'finite-draft-rank0.json'; draft = load(dp); paths.append(dp)
            keys = [k for values in draft['native_keys'].values() for k in values]
            require(draft['status'] == 'complete' and draft['explicit_keys'] == len(set(keys)) == len(keys) == 86
                    and draft['existing_native_generic_keys_retained'] == 28 and draft['original_native_flags_unchanged'] is True, 'Actual draft86+28 receipt')
            counts = dict(draft=86,generic=28,target=0); total = 114
        else:
            pp, tp = startup_dir/f'prefill-rank{rank}.json', startup_dir/f'finite-target-rank{rank}.json'
            prefill, target = load(pp), load(tp); paths += [pp,tp]
            require(prefill['status'] == 'complete' and prefill['completed'] == prefill['total'] == 201
                    and prefill['transition_checks_passed'] == 8 and target['status'] == 'complete'
                    and target['fixtures'] == len(target['native_keys']) == len(set(target['native_keys'])) == 44
                    and target['original_native_flags_unchanged'] is True and mc2['manifest_route_count'] == 128
                    and mc2['disabled_routes'] == mc2['routes_with_missing_or_unqualified_rows'] == [], 'Actual target201+44 receipt')
            counts = dict(draft=0,generic=201,target=44); total = 245
        ip = startup_dir/f'finite-inventory-rank{rank}.jsonl'
        rows = [json.loads(line) for line in ip.read_text().splitlines()]
        last = rows[-1]
        require(last['phase'] == 'V80_original_startup' and last['completed'] == last['total'] == total
                and last['memory']['graph_entries'] == total and all(row['local_errors'] == [] for row in rows), 'Actual final original startup finite checkpoint')
        paths.append(ip)
        startup.append(dict(rank=rank,actual_original_inventory_counts=counts,final_checkpoint=dict(phase=last['phase'],completed=last['completed'],total=last['total'],local_errors=last['local_errors']),
                            files=[record(path) for path in paths], HOST_reexecuted_model_gates=False))
    fusion_record = fusion(P/'fusion_result.json')
    require(any(row['sha256'] == fusion_record['sha256'] for row in r['actual_runtime_outputs']), 'Retained fusion identity')
    require(not ATTEMPTS and not any(n.split('.')[0] in FORBIDDEN for n in sys.modules), 'No backend/model imports')
    result = dict(schema='ROOT.V132.actual-failed-metadata-audit.v1',status='passed',errors=[],actual_run_status='failed',
        receipt=record(rp),script=record(__file__),actual_HOST_argv=sys.orig_argv,ROOT_pre_NPU_audit=record(pre_path),
        source_counts=source_counts,CPU=record(cpu_path),freeze=record(frozen_path),original_HOST_tests=23,
        source_archive=runtime_archive,prepared_archive=prepared_archive,
        failure=dict(classification='recursive dataclass metadata serialization failed before new qualification',source_line=88,exception_type='TypeError',message="cannot pickle 'mappingproxy' object",ranks=errors),
        startup=startup,original_startup_inventory_completed_all4=True,
        qualification=dict(V132_S35_scope_started=False,candidate_controls_installed=False,warm65_started=False,zero_started=False,seal_completed=False,full463_started=False,
            full463_completed=False,whole_workload_AR_qualified=False,formal_Goodput=False,raw_Goodput=None),
        cleanup=dict(worker_exit=record(out/'worker-exit.json'),worker_exit_values=exited,owned_remaining=r['owned_remaining'],
            all4_idle_from_original_terminal_receipt=r['devices_after']['all_idle'],physical_cards_requeried=False,pre_verifier_exit=0,post_verifier_exit=0),
        source_unchanged=dict(local_and_archived=True,external=True,actual_sourced_shells=r['actual_sourced_runtime_shells_after'],runtime_aliases=r['runtime_aliases_after'],fusion=fusion_record),
        limits=dict(attempt_private_bytes=r['attempt_private_bytes'],aggregate_private_bytes=r['aggregate_private_bytes'],storage_limits=r['storage_limits']),
        scope_gaps=['No final worker reports or completed qualification/trace artifacts exist.',
          'Startup graph flags/KV/numerics were executed by original frozen producer; this HOST audit verifies retained records and source order, without replaying model gates.',
          'No S35/live split192/rolling/CPU-clean/full-workload correctness or performance qualification follows from completed original startup.'],
        HOST_audit_prior_attempt=dict(execution=record(D/'ROOT_V132_ACTUAL_FAILED_METADATA_AUDIT_EXECUTION.json'),
            log=record(D/'ROOT_V132_ACTUAL_FAILED_METADATA_AUDIT_HOST.log'),
            original_exit_code=load(D/'ROOT_V132_ACTUAL_FAILED_METADATA_AUDIT_EXECUTION.json')['exit_code'],
            failure='HOST auditor initially assumed runtime tar naming for prepared tar; retry strips the exact inputs/ prefix, without changing accepted member set or bytes'),
        forbidden_import_attempts=ATTEMPTS,NPU_or_model_or_card_query_executed=False)
    destination = D/'ROOT_V132_ACTUAL_FAILED_METADATA_AUDIT.json'
    with destination.open('x') as stream:
        json.dump(result,stream,sort_keys=True,separators=(',',':'),allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],actual_run_status='failed',output=str(destination),sha256=sha(destination),startup_completed=True,qualification_or_full_started=False)))


if __name__ == '__main__':
    main()
