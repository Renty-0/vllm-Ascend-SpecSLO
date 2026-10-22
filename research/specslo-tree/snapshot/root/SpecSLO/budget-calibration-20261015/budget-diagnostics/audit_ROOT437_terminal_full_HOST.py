"""Source-bound terminal audit; retain completed math even on a failed job."""
import argparse, datetime, gzip, hashlib, importlib.util, json, sys
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B/'budget-diagnostics'
P = B/'tree-refill-witness-boolean-retention-full-v437'
DONOR = B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
FREEZE = '8ecd9f97279d1ae6023e4ca14188954b34575b26940ec64d1848c8b93a99f8d8'
ARCHIVE = '1553265a38fa19caf8bcb32067e9e921627f97f9c892429d68bd891e97aa366e'
OLD = D/'audit_v209_terminal_HOST.py'
OLD_SHA = 'fe8a050d3dce287787017bdf9b0f319bd2577e641e40a7baef7f2df785c0db5b'


def need(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes())



def independent_geometry(call):
    n=call['N'];budgets=call['budgets'];q=call['real_query_count']
    need(n==len(budgets)and 1<=n<=24 and all(type(v)is int and 1<=v<=5 for v in budgets)
         and sum(budgets)<=96 and q==n+sum(budgets)<=120,'actual unchanged retained-budget limits')
    need(n+budgets.count(5)<=24,'actual complete primary/side group bound')
    remaining=120-q;padding=[5]*(remaining//5)+([remaining%5]if remaining%5 else[])
    return 120,n+len(padding),[v+1 for v in budgets]+padding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--expected-receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need(args.tag and '/' not in args.tag and not args.output.exists(), 'fresh exact private terminal artifact')
    receipt_path = DONOR/f'receipt-{args.tag}.json'
    need(sha(receipt_path) == args.expected_receipt_sha256, 'exact actual terminal receipt')
    receipt = read(receipt_path)
    need(receipt['schema'] == 'ROOT.V437.best257-uniform-KV-full-own-AR-supervisor.v1'
         and receipt['status'] in ('complete', 'failed'), 'terminal425 active64 supervisor')
    need(receipt['sources_unchanged'] and not receipt['owned_remaining'] and receipt['devices_after']['all_idle'],
         'frozen source and owned cleanup, all four idle')
    for phase in ('pre_verifier', 'post_verifier'):
        evidence = receipt[phase]
        need(evidence['exit_code'] == 0 and evidence['actual_result']['status'] == 'pass'
             and sha(evidence['path']) == evidence['sha256'], 'actual original '+phase)
    need(sha(OLD) == OLD_SHA, 'unchanged independent raw arithmetic auditor')
    spec = importlib.util.spec_from_file_location('_ROOT235_original_raw_arithmetic_auditor', OLD)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    sys.meta_path.insert(0, original.Firewall())
    sys.path[:0] = [str(P), str(B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'), str(DONOR)]
    import os, shlex, re
    for line in (P/'launch437.sh').read_text().splitlines():
        for value in shlex.split(line.rstrip(' \\')):
            if re.fullmatch(r'SPECSLO_V[0-9]+_ROOT_FREEZE_SHA=[a-f0-9]{64}',value):
                key,setting=value.split('=',1);os.environ[key]=setting
    os.environ['SPECSLO_V257_ROOT_FREEZE_SHA']='b352ac6bb9f3ae182d33d2f11dc278f7e49c2b8dff9d9d514c0b32d7a14a01f2'
    startup=[v for v in (P/'startup437.sh').read_text().splitlines()if v.startswith('export PYTHONPATH=')]
    need(len(startup)==1,'one literal frozen child source path')
    value=startup[0].removeprefix('export PYTHONPATH=')
    F=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1')
    for old,new in [('$repo',str(F/'source/vllm-ascend-hust')),('$frozen',str(F)),('$task_root',str(DONOR)),('${PYTHONPATH:-}','')]:value=value.replace(old,new)
    need('$'not in value,'literal parent paths')
    sys.path[:0]=value.split(':')
    import guard437 as guard425
    guard425.EXPECTED_FREEZE = FREEZE
    guard425.EXPECTED_ARCHIVE = ARCHIVE
    _, sources, _ = guard425.prepared()
    actual = DONOR/'actual'/args.tag
    paths = [f for f in sorted(actual.rglob('*')) if f.is_file()]
    need(not any(f.is_symlink() for f in paths), 'private actual files have no alias')
    manifest = {str(f):dict(bytes=f.stat().st_size, sha256=sha(f)) for f in paths}
    result = dict(schema='ROOT437.independent-terminal-compute-audit.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        supervisor_status=receipt['status'], supervisor_errors=receipt['errors'],
        source_members=len(sources), source_freeze_sha256=FREEZE, source_archive_sha256=ARCHIVE,
        original_receipt_sha256=args.expected_receipt_sha256,
        positive_math_case_checkpoints=[len(list((actual/'canonical-independent-node-math').glob(f'rank{r}-canonical-case*.json'))) for r in range(4)],
        whole_workload_AR_qualified=False, metrics=None, errors=[],
        source_and_raw_manifest=manifest, NPU_used=False, Torch_imported=False,
        formal_Goodput=False, adopted=False)
    if all((actual/f'rank{rank}-local-diagnostic.json').exists() for rank in range(4)):
        try:
            import audit437 as audit425
            result['completed_compute_audit'] = audit425.audit(actual, (64,))
            wait=read(actual/'completed.json')['qualification_rpc_wait']
            need(wait['original_parent_timeout_seconds']==6000 and wait['qualification_receive_timeout_seconds']==18000 and wait['original_engine_configuration_object_unchanged'] is True and wait['worker_configuration_changed'] is False and wait['production_timeout_changed'] is False, 'actual unchanged worker/config with qualification-only wait')
            result['qualification_rpc_wait']=wait
            result['current75_raw_case_records'] = [len(list((actual/'parent4-current-component425/bounded199').glob(f'rank{r}-pass*.json.gz'))) for r in range(4)]
            need(result['current75_raw_case_records'] == [75]*4, 'all75 actual new-dispatcher lifecycle cases per rank')
        except Exception as error:
            result['errors'].append(dict(stage='producer_full_audit', error=repr(error)))
        try:
            original.expected_current_geometry=independent_geometry
            original.definitions(original.PARENT_AUDITOR, {'admission_and_full_statistics','common_full_statistics'}, (
                ('from empty197 import audit_trace', 'from admission_audit432 import audit_trace'),
                ("cohorts['all_arrivals_retained'] is cohorts['no_active_refill'] is True","cohorts['all_arrivals_retained'] is cohorts['source_clocks_unchanged'] is cohorts['exact_original_incumbent_prefill_charge'] is True"),
                ('empty_active_cohort_replay=cohorts','actual_admission_cohort_replay=cohorts'),
                ('actual_empty_active_only_coalesce1_0=True','actual_original_coalesce1_0=True'),
                ('no_prefill_pause_memberships_for_incumbents=True','incumbent_prefill_pause_memberships_exactly_charged=True'),
                ("trace['admission_window160']==32", "trace['admission_window160']==64"),
                ("0<trace['active_peak']<=32", "0<trace['active_peak']<=64"),
                ("if v['window'] == 32", "if v['window'] == 64"),
                ("if (c['N'], c['real_query_count'], c['family']) == (16, 80, 'causal')", "if True"),
                ("(80, 16, [v + 1 for v in c['budgets']])", "expected_current_geometry(c)"),
                ('current80_calls=len(selected80)', 'current120_calls=len(selected80)'),
                ("c['spec_rhythm_slo_aware_admission_enabled']==c['spec_rhythm_target_slo_first_enabled']==c['spec_rhythm_prefill_coalesce_enabled']==0",
                 "c['spec_rhythm_slo_aware_admission_enabled']==c['spec_rhythm_target_slo_first_enabled']==1 and c['spec_rhythm_prefill_coalesce_enabled']==0"),
                ('actual SLO0/TARGET0/original coalesce1', 'actual SLO1/LIVE_TARGET1/original coalesce1')))
            reports = [read(actual/f'rank{rank}-local-diagnostic.json') for rank in range(4)]
            traces = [read(actual/'windows/window64/full463'/f'rank{rank}-full463-actual-trace.json.gz') for rank in range(4)]
            result['metrics'] = original.admission_and_full_statistics(traces, reports)
            need(all(t['configured_capacity'] == 64 and t['admission_window160'] == 64 and 0 < t['active_peak'] <= 64 for t in traces), 'actual global64 runtime ceiling')
            peaks = [t['active_peak'] for t in traces]
            need(len(set(peaks)) == 1, 'cross-rank actual active peak')
            result['actual_active64'] = dict(configured_max_num_seqs=64, effective_active_ceiling=64, active_peak=peaks[0], peak64_observed=peaks[0] == 64, cohort_sizes=[len(row['sequence_ids']) for row in traces[1]['standalone_prefill_calls']], single_target_max_requests=24)
            target_calls = traces[1]['normal_target_calls']
            groups = [c['N'] + c['budgets'].count(5) for c in target_calls]
            need(groups and max(groups) <= 24, 'all actual complete target primary/side groups fit S24')
            result['actual_target_segment_packing425'] = dict(limit=24, peak=max(groups), calls=len(groups), branch_calls=sum(bool(c['budgets'].count(5)) for c in target_calls), source_sha256=sha(B/'tree-parent4-canonical-tail-observer-binding-v425/packing425.py'), whole_proposals_retained=True, global_active_limit=64)
            from audit_reference425 import audit as audit_reference
            result['independent_complete_own_AR'] = audit_reference(actual)
            result['whole_workload_AR_qualified'] = result['independent_complete_own_AR']['same_backend_exact_output_qualified']
            if 'completed_compute_audit' in result:
                need(result['whole_workload_AR_qualified'] == result['completed_compute_audit']['whole_workload_AR_qualified'],
                     'two complete own-history audits agree')
        except Exception as error:
            result['errors'].append(dict(stage='completed_computation', error=repr(error)))
    else:
        result['incomplete_computation'] = True
        result['retained_role_errors'] = {str(f):read(f) for f in actual.glob('rank*-last-error.json')}
    need(not original.Firewall.attempts and not any(n.split('.')[0] in original.BACKENDS for n in sys.modules),
         'HOST auditor never imported a model/device backend')
    result['status'] = 'terminal_completed_compute_audited' if result['metrics'] is not None and not result['errors'] else 'terminal_incomplete_or_audit_errors_retained'
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('status','supervisor_status','positive_math_case_checkpoints','whole_workload_AR_qualified','errors')}))


if __name__ == '__main__':
    main()
