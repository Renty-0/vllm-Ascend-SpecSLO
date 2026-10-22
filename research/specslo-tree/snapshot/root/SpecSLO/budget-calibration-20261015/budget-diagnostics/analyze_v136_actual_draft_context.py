"""Independent terminal V136 HOST audit, with safe CPU tensor witnesses."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tarfile
import traceback

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B/'budget-diagnostics'
P = B/'tree-draft-cpu-context-fixture-v136'
PRE = D/'ROOT_V136_PRE_NPU_SOURCE_AND_HOST_AUDIT.json'
PRE_SHA = 'd0574a6c7ff0732b2879c926a11154a1810ded053089875b98b846ab216715c5'
RECEIPT_SHA = 'b63d389b03a2e183ffafd984d5362147cdc0c7b87c57ca6c04fbb5732ac5d870'


def need(value, message):
    if not value:
        raise ValueError('V136 external actual: '+message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def item(path):
    return dict(path=str(path), sha256=sha(path), bytes=Path(path).stat().st_size)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError('Nonfinite JSON '+value)))


def archive(path, expected):
    with tarfile.open(path, 'r:gz') as source:
        members = source.getmembers()
        need(len(members) == len(expected) and {m.name for m in members} == set(expected),
             'exact archive coverage')
        for member in members:
            need(member.isfile(), 'regular archive members')
            stream = source.extractfile(member)
            need(hashlib.sha256(stream.read()).hexdigest() == expected[member.name],
                 'archive member byte identity')
    return dict(**item(path), members=len(expected), all_member_bytes_checked=True)


def sources(receipt):
    need(sha(PRE) == PRE_SHA, 'ROOT frozen preaudit')
    pre = read(PRE)
    cpu_path, freeze_path = P/'CPU_RESULT_V128.json', P/'PREPARED_SOURCE_HASHES_V128.json'
    cpu, frozen = read(cpu_path), read(freeze_path)
    need(pre['status'] == 'passed' and pre['readiness_exit_code'] == 0
         and pre['CPU_sha256'] == sha(cpu_path) and pre['freeze_sha256'] == sha(freeze_path),
         'ROOT actual CPU/freeze/readiness identities')
    sys.path.insert(0, str(P))
    import check_v128
    from source_snapshot_v128 import runtime_outputs
    ready = check_v128.require_ready(receipt['readiness']['identity'])
    need(cpu['tests_run'] == 17 and cpu['failures'] == cpu['errors'] == 0
         and len(frozen['files']) == 22 and len(frozen['external_files']) == 56,
         'original source/CPU scope')
    local = {**frozen['files'], str(cpu_path): sha(cpu_path), str(freeze_path): sha(freeze_path)}
    need(receipt['source_hashes'] == local and receipt['external_input_hashes'] == frozen['external_files']
         and all(sha(path) == wanted for path, wanted in {**local, **frozen['external_files']}.items()),
         'actual unchanged source/input maps')
    prepared = Path(pre['archive'])
    need(sha(prepared) == pre['archive_sha256'], 'prepared archive SHA')
    a = archive(prepared, {name.lstrip('/'): wanted for name, wanted in local.items()})
    actual = Path(receipt['source_snapshot'])
    need(actual.is_relative_to(P/'source-snapshots') and sha(actual) == receipt['source_snapshot_sha256'],
         'runtime local archive SHA/path')
    b = archive(actual, {Path(name).name: wanted for name, wanted in local.items()})
    outputs = runtime_outputs(P)
    need(outputs == receipt['runtime_outputs'], 'strict known fusion exception retained')
    return dict(passed=True, preaudit=item(PRE), CPU=item(cpu_path), freeze=item(freeze_path),
        local_sources=22, external_inputs=56, prepared_archive=a, actual_runtime_archive=b,
        external_input_archive_scope='Both archives contain local+CPU+freeze;56 external inputs are exact SHA-bound, not claimed as archived here.',
        readiness_identity=ready['identity'], fixture=ready['source_contract']['fixture_source_contract'],
        runtime_outputs=outputs)


def cleanup(receipt, root):
    need(receipt['schema'] == 'V128.actual-upstream-supervisor.v1'
         and receipt['action'] == 'DRAFT-original-CPUcontext-cost'
         and receipt['physical_cards'] == [2, 5, 6, 7] and receipt['status'] == 'complete'
         and receipt['errors'] == [] and receipt['child']['exit_code'] == 0
         and receipt['owned_remaining'] == [] and receipt['devices_before']['all_idle'] is True
         and receipt['devices_after']['all_idle'] is True
         and receipt['experimental_sources_unchanged'] is receipt['external_sources_unchanged'] is True,
         'original terminal success/owned cleanup/source checks')
    logs = []
    for side in ('pre', 'post'):
        row = receipt[side+'_verifier']
        need(row['exit_code'] == 0 and sha(row['log']) == row['log_sha256']
             and Path(row['log']).stat().st_size == row['log_bytes'], 'actual verifier '+side)
        logs.append(item(row['log']))
    paths = re.findall(r'^source (/[^ \n]+)', (B/'serial-v1/runtime.sh').read_text(), re.MULTILINE)
    shells = [dict(path=name, sha256=sha(name)) for name in paths]
    need(len(shells) == 2 and shells == receipt['actual_sourced_runtime_shells_before']
         == receipt['actual_sourced_runtime_shells_after'], 'actual CANN/ATB shell bytes')
    need(receipt['runtime_aliases_before'] == receipt['runtime_aliases_after'], 'original alias records')
    for row in receipt['runtime_aliases_after']:
        path = Path(row['path'])
        need(path.is_symlink() and os.readlink(path) == row['link']
             and str(path.resolve()) == row['resolved'] and path.exists() is row['exists'], 'actual final alias')
    exited = read(root/'worker-exit.json')
    need(exited['closed'] is exited['all_exited'] is True, 'all four workers exited')
    used = sum(path.stat().st_size for path in P.rglob('*') if path.is_file())
    need(used < 1024*1024**2 and all(path.stat().st_size <= 64*1024**2
         for path in root.iterdir() if path.is_file()), 'actual private/file limits')
    return dict(passed=True, original_elapsed_seconds=receipt['elapsed_seconds'],
        original_verifier_logs=logs, runtime_shells=shells, worker_exit=item(root/'worker-exit.json'),
        current_private_bytes=used, hardware_queried_by_this_auditor=False)


def witness_audit(draft, root):
    need(os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') == '0', 'backend autoload must be disabled')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    need(not any(name.startswith(('torch_npu', 'vllm_ascend')) for name in sys.modules),
         'no Native/NPU backend imported')
    config_path = Path('/root/data/shared-models/Qwen3-0.6B/config.json')
    config = read(config_path)
    need((config['num_hidden_layers'], config['hidden_size'], config['num_key_value_heads'], config['head_dim'])
         == (28, 1024, 8, 128), 'actual draft model tensor geometry')
    results, baseline = [], {}
    def compare(actual, reference):
        exact, maximum = True, 0.
        for a, b in zip(actual, reference, strict=True):
            need(a.dtype == b.dtype and a.shape == b.shape, 'paired tensor dtype/shape')
            need(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), 'all actual/reference tensors finite')
            need(torch.allclose(a, b, rtol=1e-3, atol=1e-3), 'original Native allclose tolerance')
            exact &= torch.equal(a, b)
            maximum = max(maximum, float((a.float()-b.float()).abs().max()))
        return dict(exact=exact, max_abs=maximum, allclose_1e3=True)
    for record in draft['qualification']:
        n, steps = (16, 4) if record['case'] == 'N16-B5-p127' else (32, 3)
        need([p['native']['runtime_validated'] for p in record['phases']] == [False, True, True]
             and [p['native_delta']['aclgraph_capture_attempts'] for p in record['phases']] == [1, 0, 0]
             and [p['native_delta']['aclgraph_runtime_validation_replays'] for p in record['phases']] == [0, 1, 0],
             'actual independent namespace cold/changed/restored Native transitions')
        for phase_index, phase in enumerate(record['phases']):
            evidence = phase['witness'];path = Path(evidence['path'])
            need(path.parent.resolve() == root.resolve() and sha(path) == evidence['sha256']
                 and path.stat().st_size == evidence['bytes'], 'actual CPU witness SHA/size')
            data = torch.load(path, map_location='cpu', weights_only=True)
            need(type(data) is dict and set(data) == {'candidate', 'independent_reference'}, 'safe witness root schema')
            for values in data.values():
                need(type(values) is dict and set(values) == {'hidden', 'encoded', 'query_KV'}, 'witness tensor families')
                need(isinstance(values['hidden'], (list, tuple)) and len(values['hidden']) == steps
                     and isinstance(values['query_KV'], (list, tuple)) and len(values['query_KV']) == 56,
                     'all physical hidden and28K/V tensor counts')
                for tensor in [*values['hidden'], *values['query_KV'], values['encoded']]:
                    need(type(tensor) is torch.Tensor and tensor.device.type == 'cpu'
                         and tensor.layout == torch.strided, 'only dense CPU tensor witness')
                need(all(tuple(t.shape) == (n, 1024) and t.dtype == torch.bfloat16 for t in values['hidden'])
                     and all(tuple(t.shape) == (n*steps, 8, 128) and t.dtype == torch.bfloat16 for t in values['query_KV'])
                     and tuple(values['encoded'].shape) == (n, steps, 2)
                     and values['encoded'].dtype == torch.int64, 'model-exact witness shape/dtype')
                encoded = values['encoded']
                need(bool((encoded >= 0).all()) and bool((encoded//1000001 < 151936).all()), 'actual encoded vocabulary')
            actual, reference = data['candidate'], data['independent_reference']
            need(torch.equal(actual['encoded'], reference['encoded']), 'full physical encoded top2 exact')
            hidden = compare(actual['hidden'], reference['hidden']);kv = compare(actual['query_KV'], reference['query_KV'])
            cross = None;key = (record['case'], phase_index)
            if record['arm'] == 'original':
                baseline[key] = actual
            else:
                previous = baseline[key]
                need(torch.equal(actual['encoded'], previous['encoded']), 'cross-arm exact full encoded')
                cross = dict(hidden=compare(actual['hidden'], previous['hidden']), KV=compare(actual['query_KV'], previous['query_KV']))
            results.append(dict(case=record['case'], arm=record['arm'], changed=phase['changed'],
                witness=item(path), hidden=hidden, all28_query_KV=kv, encoded_exact=True, cross_original=cross))
    need(len(results) == 18, 'all eighteen safe CPU witness comparisons')
    return dict(passed=True, actual_CPU_witnesses=results, model_config=item(config_path),
        torch_version=torch.__version__, CPU_threads=torch.get_num_threads(), weights_only=True,
        map_location='cpu', all18_hidden_and56_query_tensors_per_witness_recomputed_on_HOST=True,
        NPU_backend_imported=False)


def actual_records(reports, root):
    from audit_v128 import audit_reports
    original = audit_reports(reports, root)  # Exactly once in this external auditor.
    need(read(root/'actual-component-audit.json') == original, 'retained original actual audit equality')
    expected = [['calibrated', r, {'cases': 6 if r == 0 else 0, 'timings': 30}] for r in range(4)]
    need(read(root/'completed.json')['replies'] == expected, 'actual original replies')
    draft = reports[0]
    for report in reports:
        need(report['final_context']['clean'] is False and report['final_context']['observation']['passed'] is True,
             'all original CPU contexts restored')
        runtime = read(root/f"rank{report['rank']}-runtime-inputs.json")
        need(runtime['namespace']['inverse_original_AST_exact'] is True
             and runtime['namespace']['Native_math_metadata_and_validation_body_unchanged'] is True
             and runtime['target_CPU_context_changed'] is False
             and runtime['measured_target_MC2_dispatch_claimed'] is False, 'actual namespace/target scope')
        for row in [v for trace in report['zero_traces'] for v in trace]+report['cost']:
            need(row['device_completion_included'] is row['metadata_graphsubmit_D2H_CPU_pack_included']
                 is row['fixed_prefix_signature_replay'] is True
                 and row['prefix_restore_hash_context_checks_peer_wait_included'] is False, 'whole-method timer scope')
            expected_counts = dict(aclgraph_task_update_skipped_replays=int(report['is_draft']),
                aclgraph_draft_task_update_replays=0, aclgraph_draft_task_update_tasks=0)
            need(row['task_update_counter_delta'] == expected_counts, 'actual fixed-signature skip once per draft call')
    counter = draft['actual_factory_counterexample']['rows']
    need([row['output_device'].split(':')[0] for row in counter] == ['cpu', 'npu', 'cpu']
         and all(row['value'] == 7 for row in counter), 'actual omitted-device semantic counterexample')
    summaries = []
    for case in ('N16-B5-p127', 'N32-B3-p128'):
        values = {arm: [row['whole_draft_complete_ms'] for row in draft['cost']
            if row['case'] == case and row['arm'] == arm] for arm in ('original', 'CPUcontext_clean', 'original_restored')}
        need(all(len(v) == 5 and all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in v)
                 for v in values.values()), 'five actual paired costs per arm')
        medians = {arm: statistics.median(v) for arm, v in values.items()}
        for arm, median in medians.items():
            saved = next(v for v in draft['summary'] if v['case'] == case and v['arm'] == arm)
            need(saved['median_whole_draft_ms'] == median, 'independent raw median arithmetic')
        paired = []
        for sample in range(5):
            rows = {v['arm']: v['whole_draft_complete_ms'] for v in draft['cost'] if v['case'] == case and v['sample'] == sample}
            bracket = (rows['original']+rows['original_restored'])/2
            paired.append(dict(sample=sample, **rows, bracket_control_ms=bracket,
                clean_reduction_vs_bracket=1-rows['CPUcontext_clean']/bracket))
        summaries.append(dict(case=case, median_whole_method_ms=medians, paired_triplets=paired,
            median_paired_clean_reduction_vs_bracket=statistics.median(p['clean_reduction_vs_bracket'] for p in paired)))
    return dict(passed=True, original_strict_audit=original,
        original_strict_audit_canonical_sha256=hashlib.sha256(json.dumps(original, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest(), independent_cost_arithmetic=summaries,
        factory_counterexample=counter, fixed_signature_draft_skip_count=30,
        dynamic_live_task_update_qualified=False, eager_or_repair_qualified=False,
        target_MC2_dispatch_cost_measured=False, full_workload_factory_equivalence_proved=False)


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--tag', required=True);args = parser.parse_args()
    need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.tag), 'private tag')
    receipt_path = P/('receipt-'+args.tag+'.json');receipt = read(receipt_path)
    need(receipt['status'] in ('complete', 'failed'), 'terminal receipt only')
    need(sha(receipt_path) == RECEIPT_SHA, 'ROOT-notified actual receipt identity')
    output = D/'ROOT_V136_ACTUAL_DRAFT_CPU_CONTEXT_AUDIT.json';need(not output.exists(), 'exclusive audit result')
    root = P/'actual'/args.tag
    result = dict(schema='ROOT.V136.actual-draft-CPU-context-audit.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='failed', errors=[],
        actual_HOST_python=sys.executable, actual_HOST_argv=sys.orig_argv,
        environment={k: os.environ.get(k) for k in ('TORCH_DEVICE_BACKEND_AUTOLOAD', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        script=item(Path(__file__).resolve()), receipt=item(receipt_path), original_job_status=receipt['status'],
        NPU_queries=False, new_model_operations=False, full_pipeline_qualified=False,
        global_factory_equivalence_proved=False, formal_Goodput=False)
    try:
        result['source_closure'] = sources(receipt)
        result['cleanup'] = cleanup(receipt, root)
        paths = [root/f'rank{rank}-draft-context.json' for rank in range(4)]
        reports = [read(path) for path in paths];result['report_artifacts'] = [item(path) for path in paths]
        result['actual_records'] = actual_records(reports, root)
        result['safe_CPU_tensor_recomputation'] = witness_audit(reports[0], root)
        result['status'] = 'passed'
    except BaseException:
        result['errors'].append(traceback.format_exc())
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'], path=str(output), sha256=sha(output),
        error_count=len(result['errors']), full_pipeline_qualified=False, formal_Goodput=False)))
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
