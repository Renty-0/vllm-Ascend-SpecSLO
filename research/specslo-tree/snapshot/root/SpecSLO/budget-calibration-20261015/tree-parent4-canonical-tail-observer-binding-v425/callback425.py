"""Validate the actual frozen mapper-derived control instance and callback."""
import inspect
from types import CodeType


def constructor_callback(control_class):
    entries = []
    for base in control_class.__mro__:
        function = base.__dict__.get('__init__')
        if not callable(function) or not hasattr(function, '__code__'):
            continue
        entries.extend((value, function.__globals__)
                       for value in function.__code__.co_consts
                       if isinstance(value, CodeType) and value.co_name == 'candidate_shape')
    if len(entries) != 1:
        raise RuntimeError('ROOT425 one actual derived constructor callback required')
    return entries[0]


def expected_code():
    from mapper209 import controls_class
    return constructor_callback(controls_class())[0]


def inspect_actual(engine, shaper):
    import split_shape_v132
    control = getattr(engine, '_root425_controls', None)
    if control is None or control.engine is not engine or control.mode != 'candidate':
        raise RuntimeError('ROOT425 actual selected engine-owned controls required')
    method = shaper.shape.__func__
    code, _ = constructor_callback(type(control))
    reconstructed = expected_code()
    if (method is not control.candidate_shape or method.__code__ != code
            or method.__code__.co_filename != code.co_filename
            or code != reconstructed or code.co_filename != reconstructed.co_filename):
        raise RuntimeError('ROOT425 exact real control-owned derived callback code required')
    values = inspect.getclosurevars(method).nonlocals
    original = method.__globals__.get('shape')
    # The actual control owns this exact callback object. Private installation
    # clones function namespaces; code/file and policy ownership are stable,
    # whereas equality with a later sys.modules function object is not.
    if (set(values) != {'policy'} or values['policy'] is not control.policy
            or not callable(original) or original.__code__ != split_shape_v132.shape.__code__
            or original.__code__.co_filename != split_shape_v132.shape.__code__.co_filename):
        raise RuntimeError('ROOT425 original real control policy and typed split-shape code required')
    return method, control.policy, original
