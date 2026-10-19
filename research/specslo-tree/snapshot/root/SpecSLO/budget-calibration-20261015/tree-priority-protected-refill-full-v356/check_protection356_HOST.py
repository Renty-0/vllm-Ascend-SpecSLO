"""Reject real348 interruptions and real257 closed cohorts with the new audit."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys


P = Path(__file__).resolve().parent
B = P.parent
sys.path.insert(0, str(P))
import runtime356
from audit356 import check_incumbents
from empty348 import audit_trace


def load(tag):
    path = (B / 'tree-causal80-small-cohort-role-aware-audit-full-v209/actual' / tag /
            'windows/window48/full463/rank1-full463-actual-trace.json.gz')
    raw = path.read_bytes()
    trace = json.loads(gzip.decompress(raw))
    classes = {row['request_index']: row['slo_class'] for row in trace['requests']}
    if tag.startswith('static-M120-root-v257-'):
        # The old arm had its own future5/page policy. Reopen its original
        # closed-cohort audit rather than applying the newer future8 rule.
        from empty257 import audit_trace as old_closed_audit
        actual = old_closed_audit(trace)
        assert actual['no_active_refill'] and all(row['incumbent_count'] == 0 for row in actual['cohorts'])
        actual = dict(actual, actual_inflight_refill_groups=0,
                      cohorts=[dict(row, incumbent_owners=[]) for row in actual['cohorts']])
    else:
        actual = audit_trace(trace)
    return classes, actual, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def main():
    output = P / 'HOST356_PROTECTION_AUDIT_RESULT.json'
    assert not output.exists()
    records = []
    retained = []
    for tag in ('M120-B8-refill4-singleton-aware-root-v348-full463-ownAR-v1',
                'static-M120-root-v257-LIVE-target-SLO-first-full463-ownAR-v1'):
        classes, actual, source = load(tag)
        try:
            check_incumbents(classes, actual)
        except RuntimeError as error:
            records.append(dict(source=source, rejected=True, reason=str(error)))
        else:
            raise AssertionError('invalid protected admission certified')
        retained.append((classes, actual))
    classes, actual = retained[0]
    fixture = copy.deepcopy(actual)
    for group in fixture['cohorts']:
        group['incumbent_owners'] = [owner for owner in group['incumbent_owners']
                                     if classes[owner] == 'loose']
        group['incumbent_count'] = len(group['incumbent_owners'])
    fixture['actual_inflight_refill_groups'] = sum(bool(row['incumbent_count']) for row in fixture['cohorts'])
    memberships = check_incumbents(classes, fixture)
    assert memberships > 0
    assert not any(name.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend')
                   for name in sys.modules)
    value = dict(status='PASS_HOST_actual_protected_audit_rejects_retained_bad_and_closed_traces',
                 negative_controls=records, synthetic_loose_only_memberships=memberships,
                 synthetic_positive_fixture_not_full_machine_proof=True,
                 original_class_labels_and_retained_raw_files_unchanged=True,
                 NPU_used=False, full356_run_complete=False, Goodput=None)
    with output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
