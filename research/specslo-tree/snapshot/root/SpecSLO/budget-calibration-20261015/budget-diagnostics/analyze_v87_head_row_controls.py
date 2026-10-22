#!/usr/bin/env python3
"""Terminal HOST audit of fixed-reference-hidden LM-head row controls."""
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tarfile

BASE = Path('/root/SpecSLO/budget-calibration-20261015')
P = BASE/'target-head-row-control-v87'
CPU, FREEZE, RECEIPT = P/'CPU_RESULT.json', P/'SOURCE_FREEZE.json', P/'receipt.json'
FIXTURE = BASE/'budget-diagnostics/V85_NUMERICAL_MISMATCH_CASES.json'
OUT = BASE/'budget-diagnostics/ROOT_V87_ACTUAL_HEAD_ROW_CONTROL_AUDIT.json'
PYTHON = Path('/root/miniconda3/envs/vllm-hust-dev/bin/python3.11')
CPU_SHA = '9f308d04734eb9a235723985c1c85dfee726cee419f8e3b553678a3ba55e19ae'
FREEZE_SHA = 'e9a8e71cecc67a59c121fc3dc7a918dc4a755a537175d3b30374b587f5d5714f'
FIXTURE_SHA = '14d8e406920b3614413cd98cab93d67aa20f7f0b1104f1fc7bbc75e89bb3d485'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    return sha(Path(path).read_bytes())


def read(path):
    return json.loads(Path(path).read_bytes())


def verify_map(mapping, label):
    require(type(mapping) is dict and bool(mapping), label+' map absent')
    bad = [name for name,wanted in mapping.items() if not Path(name).is_absolute()
        or type(wanted) is not str or len(wanted) != 64
        or not Path(name).is_file() or file_sha(name) != wanted]
    require(not bad, label+' changed/missing: '+repr(bad[:10]))
    return dict(files=len(mapping), map_sha256=sha(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()),exact=True)


def current_local():
    return {str(path.resolve()):file_sha(path) for path in sorted(P.iterdir())
        if path.is_file() and path.suffix in ('.py','.sh','.json')
        and path.name not in ('CPU_RESULT.json','SOURCE_FREEZE.json','receipt.json')}


def verify_archive(cpu, sources, root=P):
    path = Path(cpu['actual_source_archive'])
    require(path.is_relative_to(root/'host-checks') and file_sha(path) == cpu['actual_source_archive_sha256'],
        'Actual CPU source archive identity differs')
    wanted = {Path(name).name:value for name,value in sources.items()}
    with tarfile.open(path,'r:gz') as archive:
        members = archive.getmembers()
        require(len(members) == len(wanted) == len(sources) and len({member.name for member in members}) == len(sources)
            and all(member.isfile() and member.name in wanted for member in members),
            'Actual CPU archive membership differs')
        for member in members:
            require(sha(archive.extractfile(member).read()) == wanted[member.name],
                'Actual CPU archive member bytes differ: '+member.name)
    return dict(path=str(path),sha256=cpu['actual_source_archive_sha256'],members=len(sources),exact=True)


def no_initialization(frame,event,function):
    module = frame.f_globals.get('__name__','') or ''
    name = frame.f_code.co_name
    if ((event == 'call' and module.startswith('torch_npu.npu') and name in ('init','_lazy_init'))
            or (event == 'call' and module.endswith('.native_engine') and name == '__init__')
            or (event == 'c_call' and getattr(function,'__name__','') == '_npu_init')):
        raise RuntimeError('HOST margin audit forbids model/device initialization')


def describe_profiles(terminal, case):
    """Keep every observed row; changes are findings, never parser failures."""
    base = {name:terminal[name] for name in ('full_argmax_id','greedy_id','heads_equal','known_token_scores')}
    candidate, reference = case['candidate_diff_id'], case['reference_diff_id']
    profiles = []
    for profile in terminal['head_row_profiles']:
        rows = []
        for row in profile['rows']:
            item = dict(row)
            item.update(reference_minus_candidate=row['known_token_scores'][str(reference)]-row['known_token_scores'][str(candidate)],
                full_argmax_changed_from_base=row['full_argmax_id'] != base['full_argmax_id'],
                greedy_changed_from_base=row['greedy_id'] != base['greedy_id'],
                full_argmax_is_old_candidate_id=row['full_argmax_id'] == candidate,
                greedy_is_old_candidate_id=row['greedy_id'] == candidate,
                full_argmax_flipped_from_base_to_old_candidate_id=base['full_argmax_id'] != candidate and row['full_argmax_id'] == candidate,
                greedy_flipped_from_base_to_old_candidate_id=base['greedy_id'] != candidate and row['greedy_id'] == candidate,
                known_scores_equal_base=row['known_token_scores'] == base['known_token_scores'])
            rows.append(item)
        profiles.append(dict(head_rows=profile['head_rows'],same_hidden_all_rows=profile['same_hidden_all_rows'],
            model_query_rows=profile['model_query_rows'],input_construction=profile['input_construction'],
            fullV_shape=profile['fullV_shape'],restrictedV_shape=profile['restrictedV_shape'],
            logits_dtype=profile['logits_dtype'],all_finite=profile['all_finite'],
            all_rows_retained=True,rows=rows,row0_top8=profile['row0_top8'],
            row0_equals_base={name:rows[0][name] == base[name] for name in base},
            row0_top8_equals_base=profile['row0_top8'] == terminal['top8'],
            full_argmax_candidate_rows=[row['row'] for row in rows if row['full_argmax_is_old_candidate_id']],
            greedy_candidate_rows=[row['row'] for row in rows if row['greedy_is_old_candidate_id']],
            full_argmax_flip_to_candidate_rows=[row['row'] for row in rows if row['full_argmax_flipped_from_base_to_old_candidate_id']],
            greedy_flip_to_candidate_rows=[row['row'] for row in rows if row['greedy_flipped_from_base_to_old_candidate_id']],
            heads_disagree_rows=[row['row'] for row in rows if not row['heads_equal']]))
    scope = terminal['head_row_scope']
    actual_row, actual_size = scope['actual_candidate_query_row'], scope['actual_candidate_head_rows']
    actual_profile = next(item for item in profiles if item['head_rows'] == actual_size)
    selected = None if actual_row is None else actual_profile['rows'][actual_row]
    return dict(scope=scope,profiles=profiles,all_rows_retained=True,
        M1_row_equals_base=all(profiles[0]['row0_equals_base'].values()),
        M1_top8_equals_base=profiles[0]['row0_top8_equals_base'],
        all_same_hidden_all_rows=all(item['same_hidden_all_rows'] for item in profiles),
        actual_candidate_query_row_known=actual_row is not None,
        same_reference_hidden_control_at_known_actual_candidate_row=selected,
        actual_candidate_hidden_measured=False,actual_candidate_other_rows_replayed=False)


def main():
    try:
        require(len(sys.argv) == 1 and sys.version_info[:2] == (3,11)
            and Path(sys.executable).resolve() == PYTHON.resolve() and sys.dont_write_bytecode,
            'Use retained Python3.11 -B, no arguments')
        require(not OUT.exists(),'Refusing to overwrite '+str(OUT))
        receipt_raw = RECEIPT.read_bytes()
        receipt = json.loads(receipt_raw)
        require(receipt.get('schema') == 'V87.bounded-reference-margin-diagnostic.v1'
            and receipt.get('status') in ('complete','failed'),'Supervisor is not terminal; no output created')
        paths = [P/'output'/f'rank{rank}-AR-margin-request{owner}.json' for owner in (0,1) for rank in range(4)]
        paths += [P/'output'/f'rank{rank}-margin-summary.json' for rank in range(4)]
        paths += [P/'output/completed.json',P/'output/worker-exit.json']
        require(all(path.is_file() for path in paths),'Full actual case/summary/completion/worker-exit evidence missing; no output created')
        raw = {path:path.read_bytes() for path in paths}
    except (RuntimeError,OSError,ValueError) as error:
        print('NOT RUN: '+str(error),file=sys.stderr)
        return 2
    report = dict(schema='ROOT.V87.actual-same-hidden-head-row-control-host-audit.v1',
        audited_UTC=datetime.now(timezone.utc).isoformat(),
        analysis_script=dict(path=str(Path(__file__).resolve()),sha256=file_sha(__file__)),
        evaluated_head_rows=[1,90,92,128],actual_model_query_rows_at_terminal=1,
        original_job_status=receipt['status'],original_job_status_modified=False,
        original_receipt=dict(path=str(RECEIPT),sha256=sha(receipt_raw),status=receipt['status'],
            errors=receipt.get('errors'),child_exit_code=receipt.get('child_exit_code')),
        actual_raw_sha256={str(path):sha(data) for path,data in raw.items()},
        reference_side_only=True,candidate_logits_measured=False,candidate_KV_measured=False,
        numerical_qualification=False,candidate_hidden_measured=False,actual_candidate_head_rows_replayed=False,full48_qualified=False,all65_qualified=False,
        full463_qualified=False,online_qualified=False,formal_Goodput=False,
        latency_or_Goodput_inferred=False,integrity_checks={},errors=[])
    previous = sys.getprofile()
    try:
        require(file_sha(CPU) == CPU_SHA and file_sha(FREEZE) == FREEZE_SHA
            and file_sha(FIXTURE) == FIXTURE_SHA,'Pinned CPU/freeze/fixture bytes differ')
        cpu,frozen,fixture = read(CPU),read(FREEZE),read(FIXTURE)
        require(frozen.get('schema') == 'V87.unscored-reference-margin-source.v1'
            and frozen.get('CPU_sha256') == CPU_SHA and frozen.get('online_qualified') is False
            and frozen.get('formal_Goodput') is False,'Actual freeze scope differs')
        require(cpu.get('status') == 'passed' and type(cpu.get('tests_run')) is int and cpu['tests_run'] == 13
            and cpu.get('failures') == cpu.get('errors') == 0 and cpu.get('input_unchanged') is True
            and cpu.get('forbidden_initialization_attempts') == [] and cpu.get('NPU_initialized') is False
            and cpu.get('formal_Goodput') is False,'Actual HOST13/no-model/no-NPU receipt differs')
        require(frozen['local'] == cpu['source_hashes'] == current_local() and len(frozen['local']) == 5
            and frozen['external'] == cpu['external_hashes'] and len(frozen['external']) == 1835,
            'Actual5local/1835external source closure differs')
        require(receipt.get('source_freeze_sha256') == FREEZE_SHA and receipt.get('CPU_sha256') == CPU_SHA
            and receipt.get('source_and_external_unchanged') is True and receipt.get('owned_remaining') == []
            and receipt.get('devices_after',{}).get('all_idle') is True
            and all(receipt.get(name,{}).get('exit_code') == 0 for name in ('pre_verifier','post_verifier'))
            and all(receipt.get(name) is False for name in ('formal_Goodput','full48_qualified','online_qualified')),
            'Terminal source/runtime/owned/device/scope evidence differs')
        require(type(receipt.get('errors')) is list and type(receipt.get('child_exit_code')) is int,
            'Actual terminal child/error evidence missing')
        if receipt['status'] == 'complete':
            require(receipt['errors'] == [] and receipt['child_exit_code'] == 0
                and receipt.get('local_diagnostic_execution_complete') is True,'Complete receipt is inconsistent')
        else:
            require(bool(receipt['errors']),'Failed receipt has no preserved error')
        checks = dict(local=verify_map(frozen['local'],'V87 local'),external=verify_map(frozen['external'],'V87 external'),
            CPU_source_archive=verify_archive(cpu,frozen['local']),
            CPU=dict(path=str(CPU),sha256=CPU_SHA,tests_run=13,forbidden_initialization_attempts=[]),
            freeze=dict(path=str(FREEZE),sha256=FREEZE_SHA),fixture=dict(path=str(FIXTURE),sha256=FIXTURE_SHA))
        report['integrity_checks'] = checks
        parent_binding = read(P/'PARENT_V86.json')
        parent = BASE/'target-ar-margin-diagnostic-v86'
        parent_audit_path = BASE/'budget-diagnostics/ROOT_V86_ACTUAL_REFERENCE_AR_MARGIN_AUDIT.json'
        parent_cpu_sha = '9897be678711fa17ca238d52d1331e88b6ff2088025033797974e012bedcaf1e'
        parent_freeze_sha = '135c49b7f631456c632ebd216442325b5d851bfbc3190cd1889909efdad16374'
        parent_receipt_sha = '64d6b06c6b0d436adc497e2d065b59d4bc34fb63830534d579cfb7d0b2133400'
        parent_audit_sha = 'cf92798f3bf2c12ecf831c11c7b78a1403cacaa6de8aa8b43de7fbfa66ac42f7'
        require(parent_binding['schema'] == 'V87.same-hidden-head-row-control-source-child.v1'
            and parent_binding['parent_directory'] == str(parent)
            and parent_binding['parent_freeze_sha256'] == parent_freeze_sha == file_sha(parent/'SOURCE_FREEZE.json')
            and parent_binding['parent_CPU_sha256'] == parent_cpu_sha == file_sha(parent/'CPU_RESULT.json')
            and parent_binding['parent_actual_receipt_sha256'] == parent_receipt_sha == file_sha(parent/'receipt.json')
            and parent_binding['parent_actual_audit'] == str(parent_audit_path)
            and parent_binding['parent_actual_audit_sha256'] == parent_audit_sha == file_sha(parent_audit_path)
            and parent_binding['parent_actual_job_status'] == 'complete'
            and parent_binding['reference_cases_reproduced'] == [0,1]
            and parent_binding['parent_candidate_logits_measured'] is parent_binding['parent_candidate_KV_measured'] is False
            and parent_binding['formal_Goodput'] is False,'Exact V86 parent binding differs')
        parent_audit = read(parent_audit_path)
        require(parent_audit['all_checks_passed'] is True and parent_audit['original_job_status'] == 'complete'
            and parent_audit['all_V85_reference_prefixes_reproduced'] is True
            and parent_audit['all_V85_reference_terminal_ids_reproduced'] is True
            and parent_audit['candidate_logits_measured'] is parent_audit['candidate_KV_measured'] is False,
            'Actual V86 parent reference evidence differs')
        parent_freeze, parent_cpu = read(parent/'SOURCE_FREEZE.json'), read(parent/'CPU_RESULT.json')
        require(len(parent_freeze['local']) == 4 and len(parent_freeze['external']) == 1813,
            'Exact V86 parent source extent differs')
        checks['parent_V86'] = dict(binding=parent_binding,actual_audit_sha256=parent_audit_sha,
            CPU_source_archive=verify_archive(parent_cpu,parent_freeze['local'],root=parent))
        worker_exit = json.loads(raw[P/'output/worker-exit.json'])
        require(worker_exit.get('closed') is True and worker_exit.get('all_exited') is True
            and worker_exit.get('formal_Goodput') is False,'Actual graceful worker exit is incomplete')
        report['cleanup'] = dict(owned_remaining=[],devices_after=receipt['devices_after'],
            pre_verifier=receipt['pre_verifier'],post_verifier=receipt['post_verifier'],worker_exit=worker_exit)
        sys.setprofile(no_initialization)
        sys.path.insert(0,str(P))
        actor = importlib.import_module('ar_margin_actor')
        require(Path(actor.__file__).resolve() == P/'ar_margin_actor.py','Wrong margin auditor module')
        code = compile((P/'ar_margin_actor.py').read_bytes(),str(P/'ar_margin_actor.py'),'exec',dont_inherit=True)
        for name in ('validate_fixture','recurrence_input','digest','head_row_scope','_audit_head_profiles','audit_case_report'):
            expected = [value for value in code.co_consts if type(value) is type(code) and value.co_name == name]
            require(len(expected) == 1 and getattr(actor,name).__code__ == expected[0],'Frozen pure auditor differs: '+name)
        cases = actor.validate_fixture(fixture)
        findings = []
        case_statuses = {}
        for case in cases:
            owner = case['request_index']
            observations = [json.loads(raw[P/'output'/f'rank{rank}-AR-margin-request{owner}.json']) for rank in range(4)]
            for rank,item in enumerate(observations):
                require(type(item.get('rank')) is int and item['rank'] == rank
                    and item.get('fixture_sha256') == FIXTURE_SHA
                    and item['role'] == ('draft' if rank == 0 else 'target'),'Actual role/rank/fixture binding differs')
                actor.audit_case_report(item,case)
            for key in ('completion_token_ids','calls','terminal','first_reference_difference','status'):
                require(observations[1][key] == observations[2][key] == observations[3][key],
                    'Three actual target observations disagree: '+key)
            item = observations[1]
            terminal = item['terminal']
            outputs = item['completion_token_ids']
            prefix_reproduced = outputs[:-1] == case['ref_output_prefix']
            terminal_reproduced = outputs[-1] == case['reference_diff_id']
            case_statuses[owner] = item['status']
            findings.append(dict(request_index=owner,request_id=case['request_id'],step=case['step'],
                prompt_tokens=len(case['prompt_tokens']),own_forward_calls=len(item['calls']),
                output_tokens=len(outputs),completion_sha256=actor.digest(outputs),
                actual_own_outputs=outputs,actual_own_call_trace=item['calls'],
                agreeing_target_ranks=[1,2,3],status=item['status'],
                first_reference_difference=item['first_reference_difference'],
                V85_reference_prefix_reproduced=prefix_reproduced,
                V85_reference_terminal_id_reproduced=terminal_reproduced,
                actual_terminal_input_matches_V85_reference=prefix_reproduced,
                same_hidden_heads_equal=terminal['heads_equal'],
                top8_first_id_matches_full_argmax=terminal['top8'][0]['token_id'] == terminal['full_argmax_id'],
                candidate_diff_id=case['candidate_diff_id'],reference_diff_id=case['reference_diff_id'],
                head_row_controls=describe_profiles(terminal,case),
                actual_terminal=terminal,reference_side_only=True,
                candidate_logits_measured=False,candidate_KV_measured=False))
        summaries = []
        for rank in range(4):
            summary = json.loads(raw[P/'output'/f'rank{rank}-margin-summary.json'])
            require(summary['rank'] == rank and summary['schema'] == actor.SCHEMA
                and summary['action'] == actor.ACTION and summary['fixture_sha256'] == FIXTURE_SHA
                and summary['enforce_eager'] is summary['precompile_decode_graphs'] is summary['precompile_serial_draft_graphs'] is False
                and all(summary[key] is False for key in ('candidate_logits_measured','candidate_KV_measured',
                    'numerical_qualification','online_qualified','cost_or_Goodput')),'Actual worker summary scope differs')
            require(summary['cases'] == [dict(request_index=case['request_index'],
                status='draft-coordination-only' if rank == 0 else case_statuses[case['request_index']]) for case in cases],
                'Actual summary/case statuses disagree')
            summaries.append(summary)
        completed = json.loads(raw[P/'output/completed.json'])
        require(completed['formal_Goodput'] is False and completed['full48_qualified'] is False
            and sorted(row[1] for row in completed['replies']) == list(range(4))
            and all(row[0] == 'calibrated' and row[2] == summaries[row[1]] for row in completed['replies']),
            'Actual completion replies differ from retained summaries')
        for expected,item in zip(completed['actual_reference_observations'],findings,strict=True):
            require(expected['request_index'] == item['request_index'] and expected['status'] == item['status']
                and expected['first_reference_difference'] == item['first_reference_difference']
                and expected['terminal'] == item['actual_terminal'] and expected['agreeing_target_ranks'] == [1,2,3]
                and expected['candidate_logits_measured'] is expected['candidate_KV_measured'] is False,
                'Original completed findings disagree with actual case records')
        report['actual_reference_findings'] = findings
        report['all_V85_reference_prefixes_reproduced'] = all(item['V85_reference_prefix_reproduced'] for item in findings)
        report['all_V85_reference_terminal_ids_reproduced'] = all(item['V85_reference_terminal_id_reproduced'] for item in findings)
        report['all_same_hidden_heads_equal'] = all(item['same_hidden_heads_equal'] for item in findings)
        report['all_M1_controls_equal_base'] = all(item['head_row_controls']['M1_row_equals_base'] for item in findings)
        report['all_M1_top8_equal_base'] = all(item['head_row_controls']['M1_top8_equals_base'] for item in findings)
        report['all_profile_inputs_same_hidden'] = all(item['head_row_controls']['all_same_hidden_all_rows'] for item in findings)
        report['any_full_argmax_flipped_to_old_candidate_id'] = any(profile['full_argmax_flip_to_candidate_rows']
            for item in findings for profile in item['head_row_controls']['profiles'])
        report['any_greedy_flipped_to_old_candidate_id'] = any(profile['greedy_flip_to_candidate_rows']
            for item in findings for profile in item['head_row_controls']['profiles'])
        report['interpretation'] = 'Fixed reference hidden repeated across LM-head rows only; actual candidate hidden, other rows, logits and KV were not measured; no cause of V85 mismatch established'
        sys.setprofile(previous)
        verify_map(frozen['local'],'V87 local after audit')
        verify_map(frozen['external'],'V87 external after audit')
        require(current_local() == frozen['local'] and RECEIPT.read_bytes() == receipt_raw
            and file_sha(CPU) == CPU_SHA and file_sha(FREEZE) == FREEZE_SHA
            and all(path.read_bytes() == data for path,data in raw.items())
            and file_sha(checks['CPU_source_archive']['path']) == checks['CPU_source_archive']['sha256'],
            'Original evidence changed during HOST audit')
        report['all_original_evidence_unchanged'] = True
    except Exception as error:
        report['errors'].append(type(error).__name__+': '+str(error))
    finally:
        sys.setprofile(previous)
    report['all_checks_passed'] = not report['errors']
    report['status'] = 'evidence-verified' if report['all_checks_passed'] else 'audit-rejected'
    data = (json.dumps(report,indent=2,allow_nan=False)+'\n').encode()
    descriptor = os.open(OUT,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'wb') as stream:
        stream.write(data)
    print('V87 HOST audit '+report['status']+'; original job remains '+receipt['status']+'.')
    for item in report.get('actual_reference_findings',[]):
        print('request'+str(item['request_index'])+': prefix='+str(item['V85_reference_prefix_reproduced'])
            +', terminal='+str(item['V85_reference_terminal_id_reproduced'])
            +', headsEq='+str(item['same_hidden_heads_equal'])
            +', reference-minus-candidate='+str(item['actual_terminal']['reference_minus_candidate']))
        controls = item['head_row_controls']
        print('  M1/base='+str(controls['M1_row_equals_base'])+', M1/top8='+str(controls['M1_top8_equals_base']))
        for profile in controls['profiles']:
            print('  head M'+str(profile['head_rows'])+': full flips to candidate='
                +str(profile['full_argmax_flip_to_candidate_rows'])+', greedy flips='
                +str(profile['greedy_flip_to_candidate_rows'])+', disagree rows='+str(profile['heads_disagree_rows']))
    print(str(OUT)+'. Reference-hidden LM-head row control only; no candidate hidden/logits/KV/GP qualification.')
    return 0 if report['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
