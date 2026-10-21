"""Actual typed capped3 policy, with an explicit high-active-load mode.

The original gain/need/priority/age and B3-eager body is retained. Above48
active requests, even short new normal groups use capped3 to avoid introducing
deep tickets that force the conservative target router back to S24.
"""
import hashlib
import inspect
from pathlib import Path
import sys
from types import FunctionType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       'tree-S32-actual-typed-allocation-host-v404'))
import shape404 as original
from callback409 import inspect_actual

HIGH_ACTIVE_THRESHOLD = 2 * 24


def sources():
    before = inspect.getsource(original.shape)
    edits = (
        ('    if len(admitted) <= LARGE_GROUPS:\n',
         "    if not admitted or (len(admitted) <= LARGE_GROUPS and (kwargs.get('batch_size') is None or kwargs['batch_size'] <= _HIGH_ACTIVE_THRESHOLD)):\n"),
        ('    if not LARGE_GROUPS < len(admitted) <= MAX_NORMAL:\n',
         '    if not 0 < len(admitted) <= MAX_NORMAL:\n'),
    )
    after = before
    for previous, new in edits:
        if after.count(previous) != 1:
            raise RuntimeError('ROOT409 exact404 high-load policy boundary')
        after = after.replace(previous, new)
    inverse = after
    for previous, new in reversed(edits):
        inverse = inverse.replace(new, previous)
    if inverse != before:
        raise RuntimeError('ROOT409 entire404 typed body inverse')
    return before, after


def construct():
    namespace = dict(vars(original), _HIGH_ACTIVE_THRESHOLD=HIGH_ACTIVE_THRESHOLD)
    exec(compile(sources()[1], __file__ + '::typed3-high-active-load',
                 'exec', dont_inherit=True), namespace)
    return namespace['shape']


_BODY = construct()


def shape(engine, shaper, **kwargs):
    def bind(actual_shaper, generation):
        method, policy, callback = inspect_actual(engine, actual_shaper)
        return original.Binding(method, policy, callback, generation)
    function = FunctionType(_BODY.__code__, dict(_BODY.__globals__, bind=bind),
                            _BODY.__name__, _BODY.__defaults__, _BODY.__closure__)
    return function(engine, shaper, **kwargs)


def source_descriptor():
    before, after = sources()
    return dict(original404_sha256=hashlib.sha256(before.encode()).hexdigest(),
                projected409_sha256=hashlib.sha256(after.encode()).hexdigest(),
                high_active_threshold=HIGH_ACTIVE_THRESHOLD,
                capped3_when_admitted_normal_gt24_or_active_gt48=True,
                original_split_gain_need_age_and_eager_body_preserved=True,
                complete_source_inverse=True)


def snapshot(engine):
    return dict(original.snapshot(engine), policy_projection=source_descriptor())
