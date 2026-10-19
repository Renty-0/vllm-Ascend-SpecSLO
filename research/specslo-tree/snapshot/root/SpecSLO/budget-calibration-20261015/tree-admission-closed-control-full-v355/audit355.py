"""All parent348 neural gates; explicitly require the closed-admission control."""
import inspect

import audit348 as parent


def projected_source():
    source = inspect.getsource(parent.audit)
    changes = (
        ("all(row['actual_inflight_refill_groups']>0 for row in full_refills)",
         "all(row['actual_inflight_refill_groups']==0 for row in full_refills)"),
        ('full463 must actually exercise incumbent-preserving live refills',
         'predeclared closed control must not interrupt an incumbent for refill'),
    )
    result = source
    for before, after in changes:
        if result.count(before) != 1:
            raise RuntimeError('exact parent audit admission anchor ' + before)
        result = result.replace(before, after)
    inverse = result
    for before, after in reversed(changes):
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise RuntimeError('whole parent audit source inverse')
    return result


def audit(directory, selected_windows):
    private = dict(vars(parent))
    exec(compile(projected_source(), __file__ + '::closed-control-parent-audit', 'exec'), private)
    value = private['audit'](directory, selected_windows)
    for rank in range(4):
        report = parent.json.loads((parent.Path(directory) / f'rank{rank}-local-diagnostic.json').read_text())
        if report['ROOT355_admission_mode'] != 'closed':
            raise RuntimeError('actual worker must declare closed mode')
    return dict(value, ROOT355_admission_mode='closed',
                ROOT355_parent_target_draft_and_ownAR_gates_preserved=True)
