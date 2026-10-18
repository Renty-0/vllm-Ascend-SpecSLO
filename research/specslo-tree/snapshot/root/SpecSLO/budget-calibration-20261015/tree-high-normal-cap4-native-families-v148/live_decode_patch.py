"""Source-only live fallback extension of the existing bounded decode patch.

The original builder remains available to diagnostic actors. This extension
adds reserved-prompt fallback at the two original target-only branches. It
does not install itself or authorize a formal/online workload on import.
"""
import ast
from pathlib import Path

from atomic_decode_patch import (PatchContractError, REFERENCE,
    build_blocked_runtime, replace_fresh_decode_implementation)
from diagnostic_authorization import require_bounded_scope


def _target_only_call(statement):
    return (isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == 'complete_target_only_cycle')


def build_live_runtime(source=None):
    """Add two host fallback requests to the exact original bounded source.

    Source-byte validation and all existing integration anchors belong to
    build_blocked_runtime. Only the two direct main-loop target-only branches
    may receive these additions; an unexpected count/order is rejected.
    """
    module = ast.parse(build_blocked_runtime(source))
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)
        and node.name == '_generate_spec_rhythm_tree_decode']
    if len(functions) != 1:
        raise PatchContractError('Live patch requires exactly one original decode function')
    wrappers = [node for node in functions[0].body if isinstance(node, ast.Try)]
    if len(wrappers) != 1:
        raise PatchContractError('Live patch requires the original bounded error wrapper')
    loops = [node for node in wrappers[0].body if isinstance(node, ast.While)]
    if len(loops) != 1:
        raise PatchContractError('Live patch requires exactly one original main decode loop')
    loop = loops[0]
    calls = [node for node in ast.walk(loop) if _target_only_call(node)]
    branches = [(branch, index) for branch in loop.body if isinstance(branch, ast.If)
        for index, statement in enumerate(branch.body) if _target_only_call(statement)]
    if len(calls) != 2 or len(branches) != 2 or branches[0][0] is branches[1][0]:
        raise PatchContractError('Live patch requires exactly two original target-only branches')
    for branch, index in branches:
        if index != len(branch.body)-2 or not isinstance(branch.body[index+1], ast.Continue):
            raise PatchContractError('Original target-only call must immediately precede its continue')
        branch.body[index:index] = ast.parse('''
_v75_hooks.request_original_fallback()
_v75_hooks.reset_for_original_admission_fallback(active=active,
    cache_activated=cache_activated, cache_released=cache_released,
    release_online_cache=release_online_cache)
''').body
    ast.fix_missing_locations(module)
    from runtime_adapter_v132 import adapt_live_source
    return adapt_live_source(ast.unparse(module)+'\n')


def install(native, make_hooks):
    """Explicit bounded installation, preserving the current fresh-decode shell.

    Keep its policy reset, generation, clocks and progress closure. Compile in
    a private Native namespace and replace only the wrapper's decode delegate.
    The caller supplies the live hook factory; no diagnostic factory is chosen
    implicitly here.
    """
    require_bounded_scope()
    if not callable(make_hooks):
        raise PatchContractError('Live decode installation requires a callable hook factory')
    original = native.NativePearlEngine._generate_spec_rhythm_tree_decode
    if Path(original.__globals__.get('P', '/')).resolve() != REFERENCE.parent:
        raise PatchContractError('Live decode requires the candidate original fresh_decode wrapper')
    namespace = vars(native).copy()
    namespace.update(_v75_require_local_bounded_scope=require_bounded_scope,
        _v75_make_hooks=make_hooks)
    exec(compile(build_live_runtime(), '<V81-bounded-live-mixed-prefill>', 'exec'), namespace)
    replacement = replace_fresh_decode_implementation(
        original, namespace['_generate_spec_rhythm_tree_decode'])
    native.NativePearlEngine._generate_spec_rhythm_tree_decode = replacement

