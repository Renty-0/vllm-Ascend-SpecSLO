"""Terminal-only failed-run forensic audit. No model/backend imports or queries."""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import analyze_v132_actual_failed_metadata as provenance

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B/'tree-split192-config-snapshot-fix-v138'
D = B/'budget-diagnostics'
sha, record, load, require = provenance.sha, provenance.record, provenance.load, provenance.require
ROLES = ('warmup65', 'zero65-1', 'zero65-2')
PRE_SHA = '77ea0e74a56c516933ffa16ab4c24d733a2140c2709bd75d6144a16a850914ba'


def compressed(path):
    path = Path(path)
    require(0 < path.stat().st_size <= 64*1024**2, 'Compressed artifact bound')
    with gzip.open(path, 'rb') as stream:
        raw = stream.read(256*1024**2+1)
    require(len(raw) <= 256*1024**2, 'Expanded artifact bound')
    def reject(value):
        raise ValueError('Nonfinite artifact value '+value)
    return json.loads(raw, parse_constant=reject), dict(**record(path), uncompressed_bytes=len(raw),
        uncompressed_sha256=hashlib.sha256(raw).hexdigest())


def verify_startup(receipt):
    root = Path(receipt['startup_progress']['actual_directory'])
    require(root == P/'startup-progress-by-tag'/receipt['tag'], 'Actual tag-private startup path')
    observed = []
    for rank in range(4):
        mc2_path = root/f'MC2-preflight-rank{rank}.json'
        mc2 = load(mc2_path)
        require(mc2['status'] == 'passed' and mc2['all4rank_contract_passed'] is True
            and mc2['local_passed'] is True and mc2['failed_checks'] == []
            and len(mc2['planned_full_chain_rows']) == 22
            and mc2['native_qualification_flags_modified'] is False
            and sha(mc2['profile_path']) == mc2['profile_disk_sha256'], 'Actual original startup MC2')
        files = [record(mc2_path)]
        if rank == 0:
            path = root/'finite-draft-rank0.json'; draft = load(path)
            keys = [k for values in draft['native_keys'].values() for k in values]
            require(draft['status'] == 'complete' and draft['explicit_keys'] == len(set(keys)) == len(keys) == 86
                and draft['existing_native_generic_keys_retained'] == 28
                and draft['original_native_flags_unchanged'] is True, 'Actual draft86+28')
            expected = 114; counts = dict(draft=86, generic=28, target=0); files.append(record(path))
        else:
            pp, tp = root/f'prefill-rank{rank}.json', root/f'finite-target-rank{rank}.json'
            prefill, target = load(pp), load(tp)
            require(prefill['status'] == 'complete' and prefill['completed'] == prefill['total'] == 201
                and prefill['transition_checks_passed'] == 8 and target['status'] == 'complete'
                and target['fixtures'] == len(target['native_keys']) == len(set(target['native_keys'])) == 44
                and target['original_native_flags_unchanged'] is True and mc2['manifest_route_count'] == 128
                and mc2['disabled_routes'] == mc2['routes_with_missing_or_unqualified_rows'] == [], 'Actual target201+44')
            expected = 245; counts = dict(draft=0, generic=201, target=44); files += [record(pp),record(tp)]
        path = root/f'finite-inventory-rank{rank}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        last = rows[-1]
        require(last['phase'] == 'V80_original_startup' and last['completed'] == last['total'] == expected
            and last['memory']['graph_entries'] == expected and all(row['local_errors'] == [] for row in rows), 'Actual final startup checkpoint')
        observed.append(dict(rank=rank,counts=counts,files=files+[record(path)],
            last_inventory_checkpoint=last,model_gates_reexecuted_by_HOST=False))
    return observed


def verify_completed(out, manifest):
    sys.path.insert(0,str(P))
    import audit_production_v101 as original
    from bounded_report_audit_v78 import chain_record, finite_json
    from shape_scope_v101 import validate_inventory
    from candidate_controls_v132 import MAPPER_PARTITIONS
    from target_layout import envelope as old
    from complete_search import envelope as complete
    from candidate_phases_v132 import UPDATE_FIELDS, split_delta
    from window_cost_state_v132 import validate as validate_window
    from audit_combined_v113 import branch_distribution
    checkpoints, records, signatures, chain_evidence = [], [], [], []
    for rank in range(4):
        checkpoint, checkpoint_record = compressed(out/f'rank{rank}-candidate-checkpoint.json.gz')
        require(checkpoint['rank'] == rank and [t['role'] for t in checkpoint['traces']] == list(ROLES)
            and [t['role'] for t in checkpoint['phase_control_evidence']] == list(ROLES), 'Exactly completed three-phase checkpoint')
        proof = checkpoint['mixed_same_M_qualification']
        require(proof['actual_original_actor_executed'] is proof['independent_epochs'] is True
            and proof['candidate_KV_used_as_reference'] is False, 'Independent original chain producer')
        chain = original.load_artifact(proof['artifact'],out); finite_json(chain)
        before_chain, pre_record = compressed(out/f'rank{rank}-pre-audit-same-M-qualification.json.gz')
        require(before_chain == chain, 'Retained pre-validator and post-validator complete chain differ')
        validate_inventory(chain['actual_inventory']['before'],chain['actual_inventory']['after'],is_draft=rank==0)
        signatures.append(chain_record(chain,'draft' if rank==0 else 'target',rank=rank))
        chain_evidence.append(dict(rank=rank,retained_pre_validator=pre_record,
            reported_post_validator_artifact=proof['artifact'],pre_post_complete_chain_equal=True,
            original_chain_record_reexecuted=True,original_validate_inventory_reexecuted=True,
            missing_final_worker_retention_metadata_not_synthesized=True))
        generations = [t['source_generation'] for t in checkpoint['traces']]
        require(generations == list(range(generations[0],generations[0]+3)), 'Three fresh generations')
        for state in checkpoint['phase_control_evidence']:
            for side in ('before','after'):
                controls=state[side]
                require(controls['mode']=='candidate' and controls['typed_draft192'] is True
                    and controls['CPU']['passed'] is True and controls['CPU_clean'] is True
                    and controls['normal_limit']==controls['future_eager_limit']==96
                    and controls['global_draft_limit']==192 and controls['current_target_C']==96
                    and controls['current_target_Q']==128 and controls['current_target_N']==32
                    and controls['normal_N_threshold']==8 and controls['normal_N_gt8_cap']==3
                    and controls['normal_N_le8_cap']==5 and controls['eager_exact_publish_B']==3
                    and controls['actual_effective_config']==dict(draft_token_budget=192,eager_token_cap=3,eager_reserve_tokens=32)
                    and controls['eligible_parent_B']==[3] and controls['Native_flags_assigned'] is False,
                    'Actual completed phase controls')
        checkpoints.append(checkpoint); records.append(checkpoint_record)
    require(all(value == signatures[0] for value in signatures), 'Cross-rank complete chain signature')
    require(all([t['source_generation'] for t in value['traces']] == [t['source_generation'] for t in checkpoints[0]['traces']]
        for value in checkpoints), 'Cross-rank three-generation identity')
    phases = []
    for index,role in enumerate(ROLES):
        raw = [original.load_artifact(cp['traces'][index]['artifact'],out) for cp in checkpoints]
        rows,prefix = original.manifest_rows(manifest,65)
        proof = original.audit_trace_set(raw,rows,prefix,zero=role!='warmup65')
        require(all(t['sealed_before'] is False and t['sealed_after'] is False for t in raw), 'Completed traces before unified seal')
        for call in raw[1]['normal_target_calls']:
            n,q,f = call['N'],call['real_query_count'],call['family']
            m,s,padding=(complete if (n,q,f) in MAPPER_PARTITIONS else old)(n,q,f)
            require(call['physical_M']==m and call['physical_S']==s
                and call['physical_query_partition']==[b+1 for b in call['budgets']]+list(padding),'Exact limited-six mapper')
        for cp in checkpoints:
            trace = cp['traces'][index]; window = trace['actual_window_cost_state']
            require(set(window)=={'before','after'},'Phase window snapshot pair')
            for side in ('before','after'): validate_window(window[side])
            require(window['after']['available'] is True and window['after']['source_generation']==trace['source_generation'], 'Actual generation window')
            change = trace['actual_task_update_counts']
            require(set(change)=={'before','after','delta'} and all(set(change[side])==set(UPDATE_FIELDS) for side in change)
                and all(type(change[side][k]) is int and change[side][k]>=0 for side in change for k in UPDATE_FIELDS)
                and all(change['delta'][k]==change['after'][k]-change['before'][k] for k in UPDATE_FIELDS), 'Actual update/skip delta')
            split = trace['actual_split_work_counts']
            require(split['delta']==split_delta(split['before'],split['after'])
                and all(type(v) is int and v>=0 for family in split['delta'].values() for v in family.values()), 'Original split Counter arithmetic')
            policy=split['delta']['policy']
            require(sum(policy[f'actual_draft_normal_N{n}_work_calls'] for n in range(33))==policy['actual_draft_nonempty_work_calls']
                and sum(n*policy[f'actual_draft_normal_N{n}_work_calls'] for n in range(33))==policy['actual_draft_normal_rows']
                and policy['actual_draft_normal_rows']+policy['actual_draft_eager_rows']==policy['actual_draft_work_rows'], 'Actual work row conservation')
        phases.append(dict(role=role,scope_count=65,original_audit_trace_set=proof,
            branches=branch_distribution(raw,[c['traces'][index]['actual_proposal_counts'] for c in checkpoints]),
            source_generation=raw[1]['source_generation'],artifacts=[c['traces'][index]['artifact'] for c in checkpoints],
            actual_window_cost_state=[c['traces'][index]['actual_window_cost_state'] for c in checkpoints],
            actual_task_update_counts=[c['traces'][index]['actual_task_update_counts'] for c in checkpoints],
            actual_split_work_counts=[c['traces'][index]['actual_split_work_counts'] for c in checkpoints],
            full463_or_goal_evidence=False))
    return dict(checkpoints=records,chain=chain_evidence,completed_phases=phases,
        original_four_phase_audit_not_called_or_relaxed=True,original_component_auditors_called_directly=True,
        final_worker_reports_missing=True,final_worker_metadata_not_synthesized=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    parser.add_argument('--expected-receipt-sha256',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.tag),'Tag')
    rp=P/f'local-receipt-{args.tag}.json';r=load(rp)
    require(sha(rp)==args.expected_receipt_sha256,'ROOT terminal receipt identity')
    require(r['status']=='failed' and bool(r['errors']) and r['child']['exit_code']!=0,'Actual failed terminal only')
    require(r['owned_remaining']==[] and r['devices_after']['all_idle'] is True
        and r['pre_verifier']['exit_code']==r['post_verifier']['exit_code']==0
        and r['experimental_sources_unchanged'] is r['external_sources_unchanged'] is True,'Terminal cleanup/source evidence')
    require(r['action']=='split192-candidate-full463','Original actual action')
    # A derived tag value is used only for checking a receipt whose exact bytes are ROOT-pinned.
    if 'tag' in r: require(r['tag']==args.tag,'Actual receipt tag')
    else: r=dict(r,tag=args.tag)
    cpu_path,freeze_path=P/'CPU_RESULT_V132.json',P/'PREPARED_SOURCE_HASHES_V132.json'
    cpu,frozen=load(cpu_path),load(freeze_path)
    require(cpu['status']=='passed' and cpu['tests_run']==8 and not cpu['errors'] and not cpu['failures']
        and cpu['Torch_imported'] is cpu['NPU_initialized'] is False
        and cpu['original23_suite_executed'] is False and cpu['parent23_actual_HOST_proof_verified'] is True
        and frozen['CPU_sha256']==sha(cpu_path),'Actual V138 eight HOST / parent23 reuse')
    require(cpu['input_hashes']==cpu['input_hashes_after']==frozen['files']
        and cpu['external_input_hashes']==cpu['external_input_hashes_after']==frozen['external_files'],'Actual CPU source before/after closure')
    counts=dict(local=provenance.verify_map(frozen['files']),external=provenance.verify_map(frozen['external_files']),
        runtime=provenance.verify_map(r['source_hashes']),receipt_external=provenance.verify_map(r['external_input_hashes']))
    wanted={**frozen['files'],**frozen['external_files'],str(cpu_path):sha(cpu_path),str(freeze_path):sha(freeze_path)}
    require(r['source_hashes']==wanted and r['readiness']['identity']==dict(CPU_sha256=sha(cpu_path),freeze_sha256=sha(freeze_path)), 'Actual frozen closure/readiness')
    require(sha(r['source_snapshot'])==r['source_snapshot_sha256'],'Actual runtime archive SHA')
    runtime=provenance.archive(r['source_snapshot'],wanted)
    pre_path=D/'ROOT_V138_PRE_NPU_SOURCE_AND_HOST_AUDIT.json';pre=load(pre_path)
    require(sha(pre_path)==PRE_SHA and pre['status']=='passed' and pre['CPU_sha256']==sha(cpu_path)
        and pre['freeze_sha256']==sha(freeze_path) and pre['readiness_exit_code']==0,'Actual ROOT preaudit')
    require(sha(pre['archive'])==pre['archive_sha256'],'Prepared archive SHA')
    prepared=provenance.archive(pre['archive'],wanted,member_prefix='inputs/')
    shells=re.findall(r'^source (/[^ \n]+)',(B/'serial-v1/runtime.sh').read_text(),re.MULTILINE)
    require([row['path'] for row in r['actual_sourced_runtime_shells_before']]==shells
        and r['actual_sourced_runtime_shells_before']==r['actual_sourced_runtime_shells_after']
        and all(sha(row['path'])==row['sha256'] for row in r['actual_sourced_runtime_shells_after']),'Actual two runtime shells')
    require(r['runtime_aliases_before']==r['runtime_aliases_after'],'Actual runtime aliases before/after')
    for row in r['runtime_aliases_after']:
        path=Path(row['path'])
        require(path.is_symlink() and str(path.resolve())==row['resolved']
            and path.readlink().as_posix()==row['link'] and path.exists() is row['exists'],'Runtime alias identity')
    out=P/'local-output'/args.tag
    failures=[]
    for rank in range(4):
        path=out/f'rank{rank}-last-error.json';value=load(path)
        require(value['rank']==rank and value['action']==r['action']
            and value['formal_Goodput'] is value['online_qualified'] is False
            and 'ValueError: Draft row bounds exceeded before allocation' in value['error'], 'Actual original rowbounds failure')
        failures.append(dict(**record(path),rank=rank,error=value['error']))
    exit_path=out/'worker-exit.json';exited=load(exit_path)
    require(exited==dict(closed=True,all_exited=True),'Original workers exit')
    require(not list(out.rglob('*-full463-actual-trace.json.gz')) and not list(out.glob('rank*-local-diagnostic.json')),
        'No successful full artifact or final worker report may be invented')
    evidence=verify_completed(out,Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/manifest.jsonl'))
    fusion=provenance.fusion(P/'fusion_result.json')
    require(any(row['sha256']==fusion['sha256'] for row in r['actual_runtime_outputs']),'Actual fusion output identity')
    require(not provenance.ATTEMPTS and not any(n.split('.')[0] in provenance.FORBIDDEN for n in sys.modules),'No model/backend imports')
    result=dict(schema='ROOT.V138.actual-failed-rowbounds-forensics.v1',status='passed',actual_run_status='failed',
        receipt=record(rp),script=record(__file__),provenance_helper=record(provenance.__file__),actual_HOST_argv=sys.orig_argv,
        source_counts=counts,CPU=record(cpu_path),freeze=record(freeze_path),ROOT_pre_NPU_audit=record(pre_path),
        actual_new_HOST_tests=8,parent23_reused_not_rerun=True,source_archive=runtime,prepared_archive=prepared,
        original_startup=verify_startup(r),completed_evidence=evidence,
        failure=dict(type='ValueError',message='Draft row bounds exceeded before allocation',ranks=failures,
            active_source=record(P/'split_budget_v132.py'),active_lines=[197,199],
            predicate='normal_rows>32 OR eager_rows>32 OR work_rows>64',
            exact_failing_row_counts_retained=False,exact_N_not_inferred_as_measured=True),
        full463=dict(started_from_actual_error=True,completed=False,artifact=None,rawTPS=None,Goodput=None,
            zero_discovery_for_full_verified=False,full_whole_workload_correctness_qualified=False),
        cleanup=dict(worker_exit=record(exit_path),worker_exit_values=exited,owned_remaining=r['owned_remaining'],
            actual_receipt_all4_idle=r['devices_after']['all_idle'],pre_verifier=r['pre_verifier'],post_verifier=r['post_verifier'],
            physical_cards_requeried=False),
        actual_runtime_shells=r['actual_sourced_runtime_shells_after'],runtime_aliases=r['runtime_aliases_after'],fusion=fusion,
        limits=dict(attempt_private_bytes=r['attempt_private_bytes'],aggregate_private_bytes=r['aggregate_private_bytes'],storage_limits=r['storage_limits']),
        scope_gaps=['The original four-phase whole-run auditor was not weakened or called with fake successful full reports.',
            'Three saved65-request phases and original chain gates are their own scope, not a completed463 run.',
            'Final worker report and its metadata/restore evidence were not produced; no absent field is filled as successful.',
            'Exact rowbounds failure arguments are not retained in the traceback; source-level inference is separate from measured N.',
            'No final Goodput or goal achievement follows from this failed actual.'],
        forbidden_import_attempts=provenance.ATTEMPTS,NPU_or_model_or_card_query_executed=False)
    destination=Path(args.output)
    with destination.open('x') as stream:json.dump(result,stream,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='passed',actual_run_status='failed',output=str(destination),sha256=sha(destination),
        completed_phases=list(ROLES),full463_completed=False)))


if __name__=='__main__':main()
