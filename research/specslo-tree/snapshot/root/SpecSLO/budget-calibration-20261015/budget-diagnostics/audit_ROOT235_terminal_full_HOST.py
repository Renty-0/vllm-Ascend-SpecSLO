"""Source-bound terminal audit; retain completed math even on a failed job."""
import argparse, datetime, gzip, hashlib, importlib.util, json, sys
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B/'budget-diagnostics'
P = B/'tree-canonical-contiguous-full463-own-AR-v235'
DONOR = B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
FREEZE = 'b9d62c5fc915a3fd9711e01d2f3e2e6894dffe96b28d791628a92c737cab69ca'
ARCHIVE = 'c03d778397d0ab68a2e8ec3b3353407bae593e82dc5af5e4b76bab099cfb14be'
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
    need(receipt['schema'] == 'ROOT.V235.canonical-full-own-AR-supervisor.v1'
         and receipt['status'] in ('complete', 'failed'), 'terminal235 supervisor')
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
    import guard235
    guard235.EXPECTED_FREEZE = FREEZE
    guard235.EXPECTED_ARCHIVE = ARCHIVE
    _, sources, _ = guard235.prepared()
    actual = DONOR/'actual'/args.tag
    paths = [f for f in sorted(actual.rglob('*')) if f.is_file()]
    need(not any(f.is_symlink() for f in paths), 'private actual files have no alias')
    manifest = {str(f):dict(bytes=f.stat().st_size, sha256=sha(f)) for f in paths}
    result = dict(schema='ROOT235.independent-terminal-compute-audit.v1',
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
            import audit235
            result['completed_compute_audit'] = audit235.audit(actual, (32,))
        except Exception as error:
            result['errors'].append(dict(stage='producer_full_audit', error=repr(error)))
        try:
            original.definitions(original.PARENT_AUDITOR, {'admission_and_full_statistics'}, (
                ('from empty197 import audit_trace', 'from empty209 import audit_trace'),
                ("c['spec_rhythm_slo_aware_admission_enabled']==c['spec_rhythm_target_slo_first_enabled']==c['spec_rhythm_prefill_coalesce_enabled']==0",
                 "c['spec_rhythm_slo_aware_admission_enabled']==1 and c['spec_rhythm_target_slo_first_enabled']==c['spec_rhythm_prefill_coalesce_enabled']==0"),
                ('actual SLO0/TARGET0/original coalesce1', 'actual SLO1/TARGET0/original coalesce1')))
            reports = [read(actual/f'rank{rank}-local-diagnostic.json') for rank in range(4)]
            traces = [read(actual/'windows/window32/full463'/f'rank{rank}-full463-actual-trace.json.gz') for rank in range(4)]
            result['metrics'] = original.admission_and_full_statistics(traces, reports)
            from audit_reference218 import audit as audit_reference
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
