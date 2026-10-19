"""Original348 complete gates plus observed protection of tight/normal owners."""
import inspect

import audit348 as parent


def projected_source():
    # The protected arm uses the original parent's actual-live-refill gate.
    return inspect.getsource(parent.audit)


def check_incumbents(classes, actual):
    if not actual['actual_inflight_refill_groups']:
        raise RuntimeError('protected refill must actually run with incumbents')
    checked = 0
    for group in actual['cohorts']:
        for owner in group['incumbent_owners']:
            if classes[owner] != 'loose':
                raise RuntimeError('prefill interrupted a protected tight/normal owner')
            checked += 1
    if not checked:
        raise RuntimeError('actual protected-refill membership proof required')
    return checked


def audit(directory, selected_windows):
    value = parent.audit(directory, selected_windows)
    directory = parent.Path(directory)
    for rank in range(4):
        report = parent.json.loads((directory / f'rank{rank}-local-diagnostic.json').read_text())
        if report['ROOT356_admission_mode'] != 'protect_tight_normal':
            raise RuntimeError('actual worker must declare protected admission')
    membership_count = 0
    for window in selected_windows:
        path = directory / f'windows/window{window}/full463/rank1-full463-actual-trace.json.gz'
        trace = parent.json.loads(parent.gzip.decompress(path.read_bytes()))
        from empty348 import audit_trace
        actual = audit_trace(trace)
        classes = {row['request_index']: row['slo_class'] for row in trace['requests']}
        membership_count += check_incumbents(classes, actual)
    if not membership_count:
        raise RuntimeError('actual protected-refill membership proof required')
    return dict(value, ROOT356_admission_mode='protect_tight_normal',
                ROOT356_checked_loose_incumbent_memberships=membership_count,
                ROOT356_no_tight_or_normal_owner_interrupted_by_prefill=True)
