"""Independently reread terminal finite B4 evidence; never rerun model math."""
import argparse
import hashlib
import importlib.abc
import json
from pathlib import Path
import re
import sys

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B / 'tree-parent4-complete75-time-budget-v424'
DONOR = B / 'tree-bounded-parent4-shared4-eager-diagnostic-v199'
FREEZE = '85cf1d551a6a44480a9bced1478054485acf3d731c3b465ab10725b3b4faf693'
ARCHIVE = 'c903bf0bb865c3c088dee91652d02485333c5d7cbb34a5b9f754de0485d06a7d'
BACKENDS = {'torch', 'torch_npu', 'vllm', 'vllm_ascend', 'transformers'}


class NoBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BACKENDS:
            raise RuntimeError('HOST419 attempted model backend import ' + fullname)
        return None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--expected-receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', args.tag)
    receipt_path = DONOR / f'receipt-{args.tag}.json'
    assert sha(receipt_path) == args.expected_receipt_sha256
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] != 'running' and receipt.get('finished_UTC')
    assert receipt.get('startup_dependency_repair_only') is True
    assert receipt.get('original391_numerical_bodies_unchanged') is True
    assert receipt.get('production_global64_requirement_retained') is True
    assert receipt['full463'] is False
    assert receipt['time_wrapper424'] is True and receipt['actual_runtime_limit_seconds'] == 18000
    assert receipt['complete_case_matrix'] == 75 and receipt['NN419_source_changed'] is False
    assert sha(P / 'FREEZE424.json') == FREEZE and sha(P / 'SOURCE424.tar.gz') == ARCHIVE
    sys.meta_path.insert(0, NoBackend())
    sys.path.insert(0, str(P))
    import guard424
    guard424.EXPECTED_FREEZE, guard424.EXPECTED_ARCHIVE = FREEZE, ARCHIVE
    _, sources, source_provenance = guard424.prepared()
    actual = DONOR / 'actual' / args.tag
    output = dict(schema='ROOT424.independent-terminal-finite-audit.v1',
                  receipt=str(receipt_path), receipt_sha256=sha(receipt_path),
                  supervisor_status=receipt['status'], supervisor_errors=receipt.get('errors'),
                  source_freeze_sha256=FREEZE, source_archive_sha256=ARCHIVE,
                  source_members_verified=len(sources), source_provenance=source_provenance,
                  source_unchanged=receipt.get('sources_unchanged'),
                  owned_remaining=receipt.get('owned_remaining'),
                  pre_verifier=receipt.get('pre_verifier', {}).get('actual_result'),
                  post_verifier=receipt.get('post_verifier', {}).get('actual_result'),
                  retained_role_errors={p.name: json.loads(p.read_text())
                                        for p in actual.glob('rank*-last-error.json')},
                  partial_rank0_case_records=len(list((actual / 'bounded199').glob('rank0-pass*.json.gz'))),
                  mathematical_checks_reexecuted_on_HOST=False,
                  inherited_parent_metadata_note='finite_cases15 and original391 flags describe frozen donor, new scope is N24/E4/R0..24 x3 with SLO-first and fresh source; actual audit must verify75 traces per rank',
                  NPU_used=False, full463=False, production_permission=False,
                  Raw=None, Goodput=None, adopted=False)
    if receipt['status'] == 'complete':
        assert not receipt['errors'] and receipt['actual_execution_complete'] is True
        assert receipt['sources_unchanged'] is True and receipt['owned_remaining'] == []
        assert all(output[k]['status'] == 'pass' for k in ['pre_verifier', 'post_verifier'])
        result, scope, limits = guard424.parent.actual_audit(actual)
        assert result == receipt['actual_component_audit']
        assert result['passed'] is True and result['production_permission'] is False
        assert result['full_required_natural_coverage'] is True
        assert result['finite_N'] == 24 and result['finite_E'] == 4 and result['finite_R'] == list(range(25))
        assert all(len(row['traces']) == 75 for row in result['ranks'])
        assert result['full463'] is False and result['whole_AR'] is False
        output.update(status='terminal_complete_finite_structural_audit_passed',
                      finite_audit=result, scope=scope, audit_limits=limits,
                      all_required_natural_coverage=result['full_required_natural_coverage'])
    else:
        output.update(status='terminal_incomplete_finite_evidence_retained',
                      all_required_natural_coverage=False)
    assert not any(name.split('.')[0] in BACKENDS for name in sys.modules)
    with args.output.open('x') as saved:
        json.dump(output, saved, ensure_ascii=False, indent=2)
        saved.write('\n')
    print(json.dumps({k: output[k] for k in ['status', 'supervisor_status',
                     'partial_rank0_case_records', 'all_required_natural_coverage', 'NPU_used']}))


if __name__ == '__main__':
    main()
