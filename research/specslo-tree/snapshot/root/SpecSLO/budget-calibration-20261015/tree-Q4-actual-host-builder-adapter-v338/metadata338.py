"""Bind the real retained Native HOST builder, preserving its complete body."""
from types import FunctionType, MethodType
import inspect
import metadata336 as original
from geometry336 import LAYOUTS

ensure_zero = original.ensure_zero


def inner_frontend(function):
    while hasattr(function, '__wrapped__'):
        function = function.__wrapped__
    return function


def clone_function(function, namespace, closure):
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, closure)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = dict(function.__annotations__)
    result.__dict__.update(function.__dict__)
    result.__module__ = function.__module__
    result.__qualname__ = function.__qualname__
    result.__doc__ = function.__doc__
    return result


def rebind(function, key, delegate):
    if hasattr(function, '__wrapped__'):
        child = function.__wrapped__
        private_child = rebind(child, key, delegate)
        cells = list(function.__closure__ or ())
        selected = [i for i, cell in enumerate(cells) if cell.cell_contents is child]
        if len(selected) != 1:
            raise RuntimeError('one exact original inference-decorator child cell required')
        cells[selected[0]] = (lambda value: lambda: value)(private_child).__closure__[0]
        private = clone_function(function, function.__globals__, tuple(cells))
        private.__wrapped__ = private_child
        return private
    namespace = dict(function.__globals__, **{key:delegate})
    changed = {name for name, value in function.__globals__.items() if namespace[name] is not value}
    if changed != {key}:
        raise RuntimeError('one exact private HOST metadata delegate binding')
    return clone_function(function, namespace, function.__closure__)


def install(engine):
    if engine.is_draft:
        return dict(target_only=False)
    frontend = engine.target_tree_forward.__func__
    inner = inner_frontend(frontend)
    key = '_dynamic_tree_make_host_metadata'
    if key not in inner.__code__.co_names or not callable(inner.__globals__.get(key)):
        raise RuntimeError('ROOT338 actual retained HOST builder must occur in Native frontend')
    counted = inner.__globals__[key]
    activation = original.install(engine)

    def host_builder(this, plans, roots, candidates, *args, **kwargs):
        if this is not engine or engine._q4_layout336 not in LAYOUTS:
            raise RuntimeError('same engine and explicit Q4/Q5 frame required')
        kwargs['allow_causal_fast_path'] = False
        x, positions, meta = counted(this, plans, roots, candidates, *args, **kwargs)
        width, groups = LAYOUTS[engine._q4_layout336]
        if x.numel() != width*groups or len(plans) != groups:
            raise RuntimeError('actual Native HOST input frame differs')
        return x, positions, original.canonical(engine, meta, width=width, groups=groups)

    private = rebind(frontend, key, host_builder)
    if inspect.signature(private) != inspect.signature(frontend):
        raise RuntimeError('complete original Native target signature required')
    engine.target_tree_forward = MethodType(private, engine)
    activation.update(actual_Native_HOST_builder_bound=True,
        target_frontend_code_identical=True, target_defaults_preserved=True,
        original_inference_decorators_preserved=True,
        only_target_global_rebound=key, model_callback_not_required=True)
    return activation
