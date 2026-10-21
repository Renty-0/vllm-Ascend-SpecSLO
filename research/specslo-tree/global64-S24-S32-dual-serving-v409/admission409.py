"""Project the existing class/age normal admission from32 to24 work rows.

The64 active-owner bound and32 eager-row bound are unchanged. Mutable
generation/age records remain owned by the original per-worker policy object.
"""
import hashlib
from pathlib import Path

import normal_admission_v140 as original

STATE = '_v409_normal_admission'


def sources():
    before = Path(original.__file__).read_text()
    edits = (
        ("STATE='_v140_normal_admission'", "STATE='_v409_normal_admission'"),
        ('V140', 'ROOT409'),
        ('admitted_N_histogram=[0]*33', 'admitted_N_histogram=[0]*25'),
        ('positive_grant_N_histogram=[0]*33', 'positive_grant_N_histogram=[0]*25'),
        ('if len(raw)<=32:', 'if len(raw)<=24:'),
        ('ranked[:32]', 'ranked[:24]'),
        ('len(selected.raw)>32', 'len(selected.raw)>24'),
        ('actual_work_normal_row_limit=32', 'actual_work_normal_row_limit=24'),
    )
    after = before
    for old, new in edits:
        if old not in after:
            raise RuntimeError('exact original normal admission anchor ' + old)
        after = after.replace(old, new)
    inverse = after
    for old, new in reversed(edits):
        inverse = inverse.replace(new, old)
    if inverse != before:
        raise RuntimeError('whole original normal admission source inverse')
    return before, after


def construct():
    _, source = sources()
    namespace = dict(__name__=__name__, __file__=str(Path(__file__).resolve()))
    exec(compile(source, str(Path(__file__).resolve()) + '::projected-original-admission',
                 'exec', dont_inherit=True), namespace)
    return tuple(namespace[name] for name in ['reset', 'select', 'finish', 'snapshot', '_state', 'Selection'])


reset, select, finish, snapshot, _state, Selection = construct()


def descriptor():
    before, after = sources()
    return dict(original_sha256=hashlib.sha256(before.encode()).hexdigest(),
                projected_sha256=hashlib.sha256(after.encode()).hexdigest(),
                active_owner_limit=64, normal_work_policy_limit=24, eager_hardware_limit=32,
                same_class_age_owner_order=True, original_native_home_or_KV_mapping_changed=False)
