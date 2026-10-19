"""Privately rebind one HOST delegate through the retained closure chain."""
import inspect
from types import FunctionType, MethodType

import metadata336 as original
from metadata338 import clone_function
from geometry336 import LAYOUTS

ensure_zero = original.ensure_zero
HOST_KEY = '_dynamic_tree_make_host_metadata'


def functions(root):
    found = {}
    pending = [root]
    while pending:
        function = pending.pop()
        if id(function) in found:
            continue
        found[id(function)] = function
        if len(found) > 64:
            raise RuntimeError('bounded original target closure graph required')
        pending.extend(cell.cell_contents for cell in function.__closure__ or ()
                       if isinstance(cell.cell_contents, FunctionType))
    return list(found.values())


def bind(root, delegate):
    graph = functions(root)
    targets = [f for f in graph if HOST_KEY in f.__code__.co_names
               and callable(f.__globals__.get(HOST_KEY))]
    if len(targets) != 1:
        raise RuntimeError('exactly one actual HOST builder in target closure graph')
    target = targets[0]
    memo = {}
    visiting = set()

    def project(function):
        if id(function) in memo:
            return memo[id(function)]
        if id(function) in visiting:
            raise RuntimeError('acyclic original target closure graph required')
        visiting.add(id(function))
        cells = list(function.__closure__ or ())
        changed = False
        for i, cell in enumerate(cells):
            value = cell.cell_contents
            if isinstance(value, FunctionType):
                replacement = project(value)
                if replacement is not value:
                    cells[i] = (lambda x: lambda: x)(replacement).__closure__[0]
                    changed = True
        namespace = function.__globals__
        if function is target:
            namespace = dict(namespace, **{HOST_KEY: delegate})
            changed = True
        result = clone_function(function, namespace, tuple(cells) if cells else None) if changed else function
        wrapped = getattr(function, '__wrapped__', None)
        if changed and isinstance(wrapped, FunctionType):
            result.__wrapped__ = project(wrapped)
        memo[id(function)] = result
        visiting.remove(id(function))
        return result

    private = project(root)
    for function in graph:
        replacement = memo[id(function)]
        if replacement.__code__ is not function.__code__ or inspect.signature(replacement) != inspect.signature(function):
            raise RuntimeError('all original target wrapper bodies and signatures preserved')
    return private, target, graph


def install(engine):
    if engine.is_draft:
        return dict(target_only=False)
    root = engine.target_tree_forward.__func__
    graph = functions(root)
    targets = [f for f in graph if HOST_KEY in f.__code__.co_names
               and callable(f.__globals__.get(HOST_KEY))]
    if len(targets) != 1:
        raise RuntimeError('one original Native HOST frontend required')
    counted = targets[0].__globals__[HOST_KEY]
    activation = original.install(engine)

    def host_builder(this, plans, roots, candidates, *args, **kwargs):
        if this is not engine or engine._q4_layout336 not in LAYOUTS:
            raise RuntimeError('same engine and explicit physical layout required')
        kwargs['allow_causal_fast_path'] = False
        x, positions, metadata = counted(this, plans, roots, candidates, *args, **kwargs)
        width, groups = LAYOUTS[engine._q4_layout336]
        if x.numel() != width*groups or len(plans) != groups:
            raise RuntimeError('actual Native HOST frame differs')
        return x, positions, original.canonical(engine, metadata, width=width, groups=groups)

    private, target, graph = bind(root, host_builder)
    engine.target_tree_forward = MethodType(private, engine)
    activation.update(actual_Native_HOST_builder_bound=True,
        original_inference_decorators_preserved=True,
        all_original_wrapper_code_and_signatures_preserved=True,
        target_frontend_code_identical=True, target_defaults_preserved=True,
        model_callback_not_required=True, only_target_global_rebound=HOST_KEY,
        target_closure_functions=[dict(name=f.__qualname__, file=f.__code__.co_filename)
                                  for f in graph],
        actual_HOST_body=target.__qualname__)
    return activation
