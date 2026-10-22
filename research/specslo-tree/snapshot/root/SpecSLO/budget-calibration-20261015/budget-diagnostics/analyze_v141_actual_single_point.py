#!/usr/bin/env python3
"""Read retained V141 terminal evidence on CPU; never launch/query a device.

No Torch import occurs until completion, provenance, original scalar gates and
point metadata pass. No source/receipt/report/tensor is changed. Output is new.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import heapq
import importlib.abc
import json
import math
import os
from pathlib import Path
import re
import sys
import tarfile
import traceback

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B/'tree-single-point-config-snapshot-fix-v141'
D = B/'budget-diagnostics'
ACTION = 'single-point-padding-diagnostic'
CPU_SHA = '7917ad21b61082deda645cca05f28ea096d305db9383ae7c972e3a562ca1942f'
FREEZE_SHA = '89ae85c2b7a5b3767eb8eeaa252fc2a3deac026ed601256196d68c78979de558'
ARCHIVE_SHA = 'f6beccab07d05473084bcf728925be71a8291afc36f50ff14dc76f2f316522e9'
PREAUDIT_SHA = '502e73a5283679566dcd48b36a58fb3aa90810192c54dc6b5193c076c73692aa'
SHARDS = ((0, 50646), (50646, 101292), (101292, 151938))
VOCAB = 151936
PAIRS = (7221, 7628, 8298)


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def desc(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    if path.name.endswith('.gz'):
        raw = gzip.decompress(raw)
    return json.loads(raw)


class NoDeviceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch_npu', 'acl', 'vllm', 'vllm_ascend', 'pynvml'}:
            raise RuntimeError('CPU auditor blocked device/model import: '+fullname)
        return None


def archive(path, expected):
    expected = {str(Path(name).relative_to(P)): value for name, value in expected.items()}
    with tarfile.open(path, 'r:gz') as stream:
        members = stream.getmembers()
        need(len(members) == len(expected) and len({m.name for m in members}) == len(expected)
             and all(m.isfile() for m in members) and {m.name for m in members} == set(expected),
             'exact local archive coverage / regular members')
        for member in members:
            need(hashlib.sha256(stream.extractfile(member).read()).hexdigest() == expected[member.name],
                 'archive member bytes changed: '+member.name)
    return dict(**desc(path), members=len(expected), all_member_bytes_checked=True)


def provenance(receipt):
    cp, fp = P/'CPU_RESULT_V141.json', P/'PREPARED_SOURCE_HASHES_V141.json'
    need(sha(cp) == CPU_SHA and sha(fp) == FREEZE_SHA, 'frozen CPU/freeze identity')
    cpu, freeze = read(cp), read(fp)
    need(cpu['status'] == 'passed' and cpu['preparation_epoch'] == 'V141'
         and cpu['tests_run'] == 3 and cpu['failures'] == cpu['errors'] == 0
         and cpu['parent5_HOST_proof_reused'] is True and cpu['parent5_suite_rerun'] is False
         and cpu['NPU_initialized'] is cpu['model_initialized'] is False
         and freeze['CPU_sha256'] == CPU_SHA, 'actual three HOST + parent five reused scope')
    need(cpu['input_hashes'] == cpu['input_hashes_after'] == freeze['files']
         and cpu['external_input_hashes'] == cpu['external_input_hashes_after'] == freeze['external_files'],
         'CPU before/after source maps')
    local = {**freeze['files'], str(cp): CPU_SHA, str(fp): FREEZE_SHA, **freeze['retained_attempt_files']}
    need(receipt['source_hashes'] == local and receipt['external_input_hashes'] == freeze['external_files'],
         'actual supervisor source maps')
    for name, expected in {**local, **freeze['external_files']}.items():
        need(sha(name) == expected, 'pinned source/input bytes changed: '+name)
    prepared = P/'PREPARED_SOURCE_ARCHIVE_V141.tar.gz'
    need(sha(prepared) == ARCHIVE_SHA, 'prepared archive identity')
    prepared_info = archive(prepared, local)
    snapshot = Path(receipt['source_snapshot'])
    need(snapshot.resolve().is_relative_to(P/'source-snapshots')
         and sha(snapshot) == receipt['source_snapshot_sha256'], 'actual archive path/SHA')
    actual_info = archive(snapshot, local)
    preaudit = D/'ROOT_V141_PREPARED_SOURCE_HOST_AUDIT.json'
    need(sha(preaudit) == PREAUDIT_SHA and read(preaudit)['status'] == 'passed', 'ROOT preaudit binding')
    need(receipt['actual_sourced_runtime_shells_before'] == receipt['actual_sourced_runtime_shells_after']
         and receipt['runtime_aliases_before'] == receipt['runtime_aliases_after'], 'runtime shells/aliases changed')
    for row in receipt['actual_sourced_runtime_shells_before']:
        need(sha(row['path']) == row['sha256'], 'retained CANN/ATB shell bytes')
    for side in ('pre', 'post'):
        v = receipt[side+'_verifier']
        need(v['exit_code'] == 0 and v['argv'][:4] == ['bash', str(B/'serial-v1/runtime.sh'), 'python', '-B']
             and sha(v['log']) == v['log_sha256'] and Path(v['log']).stat().st_size == v['log_bytes'],
             'retained verifier execution/log SHA')
    return dict(CPU=desc(cp), freeze=desc(fp), actual_HOST_tests=3,
                parent5_HOST_reused_not_rerun=True, local_sources=len(freeze['files']),
                external_SHA_inputs=len(freeze['external_files']), prepared_archive=prepared_info,
                actual_archive=actual_info, ROOT_preaudit=desc(preaudit),
                local_source_bytes_archived=True, external_inputs_SHA_bound_not_all_archived=True,
                runtime_import_event_history_available=False)


def terminal(receipt, root, tag):
    need(receipt['status'] == 'complete' and receipt['errors'] == []
         and receipt['local_execution_complete'] is True and receipt['child']['exit_code'] == 0
         and receipt['owned_remaining'] == [] and receipt['physical_cards'] == [2, 5, 6, 7]
         and receipt['experimental_sources_unchanged'] is receipt['external_sources_unchanged'] is True,
         'complete successful terminal / owned cleanup / frozen sources')
    need(receipt['action'] == ACTION and receipt['child']['argv'] == [
        'bash', str(P/'launch_diagnostic_v139.sh'), '--action', ACTION, '--tag', tag], 'actual launcher argv')
    for side in ('before', 'after'):
        state = receipt['devices_'+side]
        need(state['all_idle'] is state['four_910B2'] is True and state['hardware']['exit_code'] == 0
             and state['hardware']['timed_out'] is False and set(state['processes']) == {'2', '5', '6', '7'}
             and all(v['exit_code'] == 0 and v['idle'] is True for v in state['processes'].values()),
             'retained pre/post idle snapshots')
    expected = [['calibrated', r, dict(rows=0, timings=0)] for r in range(4)]
    need(read(root/'completed.json')['action'] == ACTION
         and read(root/'completed.json')['replies'] == receipt['completed_replies'] == expected
         and read(root/'worker-exit.json') == dict(closed=True, all_exited=True), 'completion / worker exit')
    return dict(retained_pre_post_idle=True, owned_remaining=[], completion=desc(root/'completed.json'),
                worker_exit=desc(root/'worker-exit.json'))


def native_zero(value, names):
    need(set(value['before']) == set(value['after']) == set(value['delta']) == set(names)
         and all(type(v) is int and v >= 0 for side in ('before', 'after') for v in value[side].values())
         and {k: value['after'][k]-value['before'][k] for k in names} == value['delta']
         and not any(value['delta'].values()), 'explicit all21 zero delta arithmetic')


def metadata(reports, source_map, names):
    point = reports[1]['point']
    for report in reports:
        rank = report['rank']
        need(report['actual_physical_cards'] == [2, 5, 6, 7]
             and report['physical_rank_card'] == [2, 5, 6, 7][rank]
             and report['actual_target_leader_rank'] == 1, 'physical/rank/leader binding')
        for row in report['actual_runtime_origins'].values():
            need(source_map.get(row['actual_file']) == row['sha256'], 'actual helper __file__/SHA source binding')
        expected_native = dict.fromkeys(names, 0)
        if rank:
            for name in ('aclgraph_captures', 'aclgraph_capture_attempts',
                         'aclgraph_runtime_validation_replays', 'aclgraph_target_runtime_validation_calls'):
                expected_native[name] = 4
        need(report['final_Native'] == expected_native, 'minimal four-key naturally validated Native inventory')
        if not rank:
            continue
        need({(q['mode'], q['owner_order']) for q in report['qualification']} ==
             {(m, o) for m in ('old', 'new') for o in ('canonical', 'reverse')}, 'exact qualification arms')
        for q in report['qualification']:
            for kind in ('greedy', 'hidden'):
                existing = q[kind+'_existing_before']
                need([v['runtime_validated_after'] for v in q[kind+'_phases']] ==
                     ([True]*3 if existing else [False, True, True]), 'natural cold/changed/restored flags')
                for phase in q[kind+'_phases']:
                    need(phase['prefix_before'] == phase['prefix_after'], 'qualification prefix guard')
        for z in report['zero_traces']:
            for row in z:
                native_zero(row['Native'], names)
        for row in report['observations']:
            native_zero(row['Native'], names)
            c = row['capture']['context']
            need((c['ordinal'], c['mode'], c['M'], c['S'], c['production_head_M'], c['point']) ==
                 (row['ordinal'], row['mode'], row['M'], row['S'], row['M'], point)
                 and c['projection_recomputed'] is False and c['same_actual_prefix_bank'] == 'diagnostic-common-bank'
                 and c['prefix_before'] == row['prefix_before'], 'same original full-M projection / actual bank')
            plans = c['full_physical_plans']; offset = 0; selected = None
            for plan in plans:
                need(plan['first_model_row'] == offset and plan['query_rows'] == plan['candidate_budget']+1
                     and len(plan['candidate_tokens']) == plan['candidate_budget'], 'complete physical plan row mapping')
                if offset <= 43 < offset+plan['query_rows']:
                    selected = (plan, 43-offset)
                offset += plan['query_rows']
            need(len(plans) == row['S'] and offset == row['M'] and selected is not None, 'physical M/S closure')
            plan, q = selected
            need(plan['sequence_id'] == 10 and plan['real_owner'] is True and q == 3
                 and plan['candidate_budget'] == 3 and plan['parent_indices'] == [-1, 0, 1]
                 and plan['positions'][q] == 133 and ([plan['root']]+plan['candidate_tokens'])[q] == 65
                 and c['expected_slot'] == row['capture']['rows'][0]['actual_slot'], 'owner10/q3 actual data alignment')
    return dict(original_full_M_heads=[165, 128, 165], identity_head_and_model_row=43,
                point=point, per_rank_hidden_sampled_not_assumed_equal=True,
                per_layer_hidden_missing=True, all_prefix_KV_tensor_bytes_missing=True)


def numerical(reports, root):
    # Enforced before importing Torch, so running with an unsafe shell fails closed.
    need(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))
         and os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') == '0', 'required CPU thread/autoload environment')
    import torch
    torch.set_num_threads(1)
    need(torch.get_num_threads() == 1 and not any(k.startswith('torch_npu') for k in sys.modules), 'CPU-only Torch')
    loaded = {}; paths = set(); artifacts = []
    for report in reports[1:]:
        for row in report['observations']:
            path = Path(row['raw_artifact']['path'])
            need(path.resolve().is_relative_to(root/'tensors') and path not in paths, 'unique tensor path in this attempt')
            paths.add(path); artifact = desc(path); declared = row['raw_artifact']
            need(artifact['bytes'] == declared['bytes'] and artifact['sha256'] == declared['sha256'], 'tensor artifact SHA/bytes')
            def original_cpu_storage(storage, location):
                need(location == 'cpu', 'artifact was saved with a non-CPU storage tag')
                return storage
            tensors = torch.load(path, map_location=original_cpu_storage, weights_only=True)
            need(type(tensors) is dict and set(tensors) == set(row['tensor_manifest']), 'exact tensor names')
            for name, tensor in tensors.items():
                info = row['tensor_manifest'][name]
                need(type(tensor) is torch.Tensor and tensor.device.type == 'cpu' and tensor.dtype == torch.bfloat16
                     and list(tensor.shape) == info['shape'] and bool(torch.isfinite(tensor).all())
                     and hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() == info['sha256'],
                     'actual finite CPU BF16 raw tensor identity: '+name)
            loaded[report['rank'], row['ordinal']] = tensors
            artifacts.append(dict(rank=report['rank'], ordinal=row['ordinal'], **artifact))
    need(len(paths) == 9 and sum(p.stat().st_size for p in (root/'tensors').glob('*.pt')) <= 8*1024**2,
         'nine point artifacts / aggregate8MiB tensor cap')

    def stats(a, b):
        delta = a.float()-b.float()
        return dict(raw_bytes_equal=bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))),
                    torch_equal=bool(torch.equal(a, b)), original_matcher=bool(torch.allclose(a, b, rtol=1e-3, atol=1e-3)),
                    max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt()))

    comparisons = {}
    for rank in (1, 2, 3):
        comparisons[rank] = {}
        for right, label in ((1, 'old_to_new'), (2, 'old_to_old_restore')):
            a, b = loaded[rank, 0], loaded[rank, right]
            original = reports[rank]['comparison'][label]
            kv = [dict(layer=l, kind=k, **stats(a[f'KV/{l}:{k}'], b[f'KV/{l}:{k}']))
                  for l in range(64) for k in ('key', 'value')]
            result = dict(hidden=stats(a['hidden'], b['hidden']), local_logits=stats(a['local_logits'], b['local_logits']),
                current_position_KV=kv, all64_current_position_KV_original_matcher=all(v['original_matcher'] for v in kv),
                all128_KV_raw_bytes_equal=all(v['raw_bytes_equal'] for v in kv),
                first_raw_byte_differing_KV_layer=next((v['layer'] for v in kv if not v['raw_bytes_equal']), None),
                first_torch_equal_differing_KV_layer=next((v['layer'] for v in kv if not v['torch_equal']), None),
                first_original_matcher_failing_KV_layer=next((v['layer'] for v in kv if not v['original_matcher']), None))
            for name in ('hidden', 'local_logits'):
                need(result[name]['torch_equal'] == original[name]['bit_exact']
                     and result[name]['original_matcher'] == original[name]['original_matcher']
                     and result[name]['max_abs'] == original[name]['max_abs'], 'retained original point scalar mismatch')
                result[name]['producer_rms'] = original[name]['rms']
            need(result['all64_current_position_KV_original_matcher'] == original['all64_current_position_KV_original_matcher']
                 and result['first_torch_equal_differing_KV_layer'] == original['first_bit_differing_KV_layer']
                 and result['first_original_matcher_failing_KV_layer'] == original['first_original_matcher_failing_KV_layer'],
                 'retained aggregate KV matcher/first-layer mismatch')
            for new, old in zip(kv, original['current_position_KV'], strict=True):
                need((new['layer'], new['kind'], new['torch_equal'], new['original_matcher'], new['max_abs']) ==
                     (old['layer'], old['kind'], old['bit_exact'], old['original_matcher'], old['max_abs']), 'perlayer retained KV scalar mismatch')
            comparisons[rank][label] = result

    global_rows = []
    for ordinal in range(3):
        scores = []
        for rank, (start, end) in zip((1, 2, 3), SHARDS, strict=True):
            row = reports[rank]['observations'][ordinal]
            need((row['local_vocab_start'], row['local_vocab_end'], row['effective_local_vocab_end']) ==
                 (start, end, min(end, VOCAB)), 'actual shard ownership / pad boundary')
            values = loaded[rank, ordinal]['local_logits'][:min(end, VOCAB)-start].float().tolist()
            local = heapq.nsmallest(2, range(len(values)), key=lambda i: (-values[i], start+i))
            top = [dict(token_id=start+i, score=values[i]) for i in local]
            need(top == row['local_top2'] and row['local_max_tie_ids'] ==
                 [start+i for i, value in enumerate(values) if value == top[0]['score']], 'raw local top2/ties')
            for pair in row['pair_scores']:
                owned = start <= pair['token_id'] < min(end, VOCAB)
                need(pair['owned_by_rank'] is owned and pair['score'] ==
                     (values[pair['token_id']-start] if owned else None), 'raw pair score/ownership')
            scores.extend(values)
        need(len(scores) == VOCAB, 'all151936 actual full-vocab scores, final two pads excluded')
        top_ids = heapq.nsmallest(2, range(VOCAB), key=lambda i: (-scores[i], i))
        ties = [i for i, value in enumerate(scores) if value == scores[top_ids[0]]]
        need(all(reports[r]['observations'][ordinal]['selected_token'] == top_ids[0] for r in (1, 2, 3)),
             'raw full-vocab max / lowest-ID tie agrees with original threeTP production token')
        global_rows.append(dict(ordinal=ordinal, mode=('old', 'new', 'old')[ordinal], M=(165, 128, 165)[ordinal],
            production_token=top_ids[0], top2=[dict(token_id=i, score=scores[i]) for i in top_ids],
            top2_margin=scores[top_ids[0]]-scores[top_ids[1]], maximum_tie_ids=ties,
            pair_scores={i: scores[i] for i in PAIRS}, full_vocab_scores=VOCAB,
            logits_source='actual original full-M production head; no model/head recomputation'))
    restore = {rank: dict(hidden_raw_bitexact=comparisons[rank]['old_to_old_restore']['hidden']['raw_bytes_equal'],
                         logits_raw_bitexact=comparisons[rank]['old_to_old_restore']['local_logits']['raw_bytes_equal'],
                         all64_KV_raw_bitexact=comparisons[rank]['old_to_old_restore']['all128_KV_raw_bytes_equal'])
               for rank in (1, 2, 3)}
    tp_hidden = [dict(ordinal=i, rank_pair=[left, right],
                      **stats(loaded[left, i]['hidden'], loaded[right, i]['hidden']))
                 for i in range(3) for left, right in ((1,2),(1,3),(2,3))]
    return dict(global_production=global_rows, point_comparisons_by_rank=comparisons,
                actual_TP_hidden_comparisons=tp_hidden,
                old_restored_bitexact_separate=restore, tensor_artifacts=artifacts,
                actual_tensors=1170, actual_torch_version=torch.__version__, actual_torch_threads=torch.get_num_threads(),
                statistics_scope='FP32 CPU maxabs/RMS, original BF16 allclose(atol=rtol=1e-3), torch.equal, raw bytes reported separately; producer RMS reductions may differ in summation order',
                protocol_gate_weakened=False)


def run(receipt_path, tag):
    need(sys.version_info[:2] == (3, 11), 'retained Python3.11 required')
    receipt_path = Path(receipt_path).resolve()
    need(receipt_path == P/f'receipt-{tag}.json', 'explicit V141 receipt/tag identity')
    receipt = read(receipt_path)
    need(receipt.get('status') == 'complete', 'terminal complete receipt required; do not analyze running/failed raw')
    root = P/'actual'/tag
    originals = {str(p): sha(p) for p in root.rglob('*') if p.is_file()}
    originals[str(receipt_path)] = sha(receipt_path)
    guard = NoDeviceImports(); sys.meta_path.insert(0, guard); sys.path[:0] = [str(P), str(B/'tree-qualified44-cost-v114')]
    try:
        clean = terminal(receipt, root, tag); prov = provenance(receipt)
        from audit_point_v139 import audit_reports
        from mapper_scope_v135 import NATIVE_COUNTERS
        from mc2_inventory_contract import audit as audit_mc2
        reports = [read(root/f'rank{rank}-single-point.json.gz') for rank in range(4)]
        need(receipt['report_paths'] == [str(root/f'rank{rank}-single-point.json.gz') for rank in range(4)], 'exact report paths')
        scalar = audit_reports(reports, private_directory=root)
        mc2 = audit_mc2(P/'startup-progress')
        need(all(r['MC2_static'] == read(P/'startup-progress'/f'MC2-preflight-rank{r["rank"]}.json') for r in reports), 'actual startup/report equality')
        freeze = read(P/'PREPARED_SOURCE_HASHES_V141.json')
        context = metadata(reports, {**freeze['files'], **freeze['external_files']}, NATIVE_COUNTERS)
        need('torch' not in sys.modules, 'Torch must remain unloaded until all retained metadata gates pass')
        values = numerical(reports, root)
        need(values['global_production'] == [dict(**row, M=(165,128,165)[i], full_vocab_scores=VOCAB,
             logits_source='actual original full-M production head; no model/head recomputation')
             for i, row in enumerate(scalar['global_production'])], 'raw full-vocab statistics agree with frozen scalar auditor')
        need(all(sha(path) == value for path, value in originals.items()), 'actual evidence/receipt changed during read-only analysis')
        return dict(status='passed', errors=[], receipt=desc(receipt_path), tag=tag,
            source_provenance=prov, cleanup=clean, original_scalar_auditor_passed=True, MC2_startup=mc2,
            point_metadata=context, reports=[desc(root/f'rank{r}-single-point.json.gz') for r in range(4)], **values,
            independent_sameM_full_hidden_and_queryKV_bytes_retained=False,
            original_sameM_device_gate_markers_rechecked=True,
            crossM_all_query_tokens_equal=scalar['crossM_original_tokens_equal'],
            diagnostic_point_only=True, teacher_forced_not_own_AR=True, unique_numeric_or_attention_cause_proved=False,
            NZ14_unique_cause_proved=False, extra4_adoption=False, full463_qualified=False, formal_Goodput=False,
            NPU_initialized=False, model_initialized=False, NPU_query_or_runtime_launched=False)
    finally:
        sys.meta_path.remove(guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.tag), 'private tag syntax')
    output = Path(args.output).resolve()
    need(output.parent == D and not output.exists(), 'new external diagnostic output only')
    started = datetime.now(timezone.utc).isoformat()
    try:
        result = run(args.receipt, args.tag)
    except BaseException:
        result = dict(status='audit_failed', errors=[traceback.format_exc()], formal_Goodput=False,
                      NPU_initialized=False, model_initialized=False, NPU_query_or_runtime_launched=False)
    result.update(schema='V141.external-CPU-single-point-analysis.v1', actual_argv=sys.argv,
                  actual_python=sys.executable, runtime_wrapper_executed_by_auditor=False,
                  caller_runtime_wrapper_history='not inferred; usage invokes retained Python directly',
                  actual_thread_environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TORCH_DEVICE_BACKEND_AUTOLOAD')},
                  started_UTC=started, finished_UTC=datetime.now(timezone.utc).isoformat(), auditor_source=desc(__file__))
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=result['status'], output=str(output), errors=result['errors'])))
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
