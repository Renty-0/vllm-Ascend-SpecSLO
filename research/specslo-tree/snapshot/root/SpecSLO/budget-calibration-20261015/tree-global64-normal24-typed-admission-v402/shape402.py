"""Use the actual typed V148 allocator with only normal admission capped24."""
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import textwrap

import admission402
import normal_admission_v140

BASE = Path('/root/SpecSLO/budget-calibration-20261015/tree-high-normal-cap4-native-families-v148')


@dataclass(frozen=True)
class Binding:
    method: object
    policy: object
    original_shape: object
    projected_shape: object
    original_sha256: str
    projected_sha256: str


def binding(shaper):
    method = shaper.shape.__func__
    if Path(method.__code__.co_filename) != BASE / 'candidate_controls_v132.py':
        raise RuntimeError('ROOT402 actual typed CandidateControls shaper callback required')
    closure = inspect.getclosurevars(method).nonlocals
    policy = closure.get('policy')
    original = method.__globals__.get('shape')
    if policy is None or not callable(original) or Path(original.__code__.co_filename) != BASE / 'split_shape_v132.py':
        raise RuntimeError('ROOT402 original policy closure and exact split shape required')
    before = textwrap.dedent(inspect.getsource(original))
    old, new = 'if len(raw_normal)>32:', 'if len(raw_normal)>24:'
    if before.count(old) != 1:
        raise RuntimeError('ROOT402 one exact original typed admission threshold')
    after = before.replace(old, new)
    if after.replace(new, old) != before:
        raise RuntimeError('ROOT402 complete typed-shape source inverse')
    namespace = dict(original.__globals__, select_normal=admission402.select, finish_normal=admission402.finish)
    exec(compile(after, str(Path(__file__).resolve()) + '::same-typed-allocator-normal24',
                 'exec', dont_inherit=True), namespace)
    return Binding(method, policy, original, namespace['shape'],
                   hashlib.sha256(before.encode()).hexdigest(),
                   hashlib.sha256(after.encode()).hexdigest())


def shape(engine, shaper, **kwargs):
    actual = getattr(engine, '_normal402_binding', None)
    if actual is None:
        actual = binding(shaper)
        engine._normal402_binding = actual
    if shaper.shape.__func__ is not actual.method:
        raise RuntimeError('ROOT402 original shaper callback changed')
    generation = getattr(engine, '_dynamic_tree_generation', None)
    legacy = normal_admission_v140._state(actual.policy)
    if type(generation) is not int or generation <= 0 or legacy['generation'] != generation:
        raise RuntimeError('ROOT402 exact real existing phase generation required')
    current = getattr(actual.policy, admission402.STATE, None)
    if current is None or current['generation'] != generation:
        admission402.reset(actual.policy, generation)
    return actual.projected_shape(actual.policy, shaper, **kwargs)


def snapshot(engine):
    actual = getattr(engine, '_normal402_binding', None)
    if actual is None:
        return dict(available=False, global_active_limit=64, normal_work_limit=24)
    return dict(available=True, global_active_limit=64, normal_work_limit=24,
                original_typed_shape_sha256=actual.original_sha256,
                projected_typed_shape_sha256=actual.projected_sha256,
                normal_admission=admission402.snapshot(actual.policy),
                original_gain_model_and_paid_role_limits_unchanged=True,
                same_original_policy_object=True, Native_flags_assigned=False)
