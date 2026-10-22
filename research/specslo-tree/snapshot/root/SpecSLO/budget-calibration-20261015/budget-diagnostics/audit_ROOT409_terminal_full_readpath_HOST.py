"""Source-bound terminal audit; retain completed math even on a failed job."""
import argparse, datetime, gzip, hashlib, importlib.util, json, sys
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B/'budget-diagnostics'
P = B/'tree-global64-S24-S32-dual-serving-v409'
DONOR = B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
FREEZE = '62018730f2831dca46d42957eab3f4106671866f69b34b42c40bda5e2d8c3ce0'
ARCHIVE = '8d622dda7e3103326ff79cf611529a507a6c7f04dfc62fd074863eef7ce3f919'
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
    need(n==len(budgets)and 1<=n<=32 and all(type(v)is int and 1<=v<=5 for v in budgets)
         and sum(budgets)<=96 and q==n+sum(budgets)<=120,'actual unchanged retained-budget limits')
    if n > 24:
        need(max(budgets)<=3 and sum(budgets)<=88 and budgets.count(3)<=24,'wide N32 C88 B3 quota')
    else:
        need(n+budgets.count(5)<=24,'narrow complete primary/side group bound')
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
    need(receipt['schema'] == 'ROOT.V409.best257-uniform-KV-full-own-AR-supervisor.v1'
         and receipt['status'] in ('complete', 'failed'), 'terminal409 active64 supervisor')
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
    # HOST-only reader uses the exact frozen child module search paths.
    # The original failed supervisor and frozen model source stay unchanged.
    line = next(v for v in (P/'startup409.sh').read_text().splitlines()
                if v.startswith('export PYTHONPATH='))
    paths = line.removeprefix('export PYTHONPATH=')
    for before, after in [('$repo', '/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust'),
                          ('$frozen', '/root/data/specslo-baselines/nano-pearl-ascend-v1'),
                          ('$task_root', str(DONOR)), ('${PYTHONPATH:-}', '')]:
        paths = paths.replace(before, after)
    need('$' not in paths, 'literal retained HOST reader paths')
    sys.path[:0] = [v for v in paths.split(':') if v]
    import guard409 as guard409
    guard409.EXPECTED_FREEZE = FREEZE
    guard409.EXPECTED_ARCHIVE = ARCHIVE
    _, sources, _ = guard409.prepared()
    actual = DONOR/'actual'/args.tag
    paths = [f for f in sorted(actual.rglob('*')) if f.is_file()]
    need(not any(f.is_symlink() for f in paths), 'private actual files have no alias')
    manifest = {str(f):dict(bytes=f.stat().st_size, sha256=sha(f)) for f in paths}
    result = dict(schema='ROOT409.independent-terminal-compute-audit.v1',
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
            import audit409
            result['completed_compute_audit'] = audit409.audit(actual, (64,))
        except Exception as error:
            result['errors'].append(dict(stage='producer_full_audit', error=repr(error)))
        try:
            original.expected_current_geometry=independent_geometry
            original.definitions(original.PARENT_AUDITOR, {'admission_and_full_statistics','common_full_statistics'}, (
                ('from empty197 import audit_trace', 'from empty409 import audit_trace'),
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
            result['actual_active64'] = dict(configured_max_num_seqs=64, effective_active_ceiling=64, active_peak=peaks[0], peak64_observed=peaks[0] == 64, cohort_sizes=[len(row['sequence_ids']) for row in traces[1]['standalone_prefill_calls']], single_target_max_requests=32)
            target_calls = traces[1]['normal_target_calls']
            groups = [c['N'] + c['budgets'].count(5) for c in target_calls]
            need(groups and all((c['N']>24 and c['N']<=32 and max(c['budgets'])<=3 and sum(c['budgets'])<=88 and c['budgets'].count(3)<=24) or (c['N']<=24 and group<=24) for c,group in zip(target_calls,groups,strict=True)), 'every actual dual target request and group quota')
            wide_count = sum(c['N']>24 for c in target_calls)
            need(wide_count > 0, 'new route used in actual formal463 traffic')
            result['actual_target_segment_packing409'] = dict(limit=32, narrow_limit=24, wide_C88=True, wide_calls=wide_count, peak=max(groups), calls=len(groups), branch_calls=sum(bool(c['budgets'].count(5)) for c in target_calls), source_sha256=sha(P/'packing409.py'), whole_proposals_retained=True, global_active_limit=64)
            from audit_reference409 import audit as audit_reference
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
