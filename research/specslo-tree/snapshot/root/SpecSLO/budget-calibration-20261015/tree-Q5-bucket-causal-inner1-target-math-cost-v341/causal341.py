"""Keep eager/capture and task refresh on the same causal inner_precise ABI."""
import hashlib
import __future__
import inspect
import textwrap
from types import FunctionType


def project(function):
    source = textwrap.dedent(inspect.getsource(function))
    old = 'args.update(sparse_mode=3, next_tokens=0)'
    new = 'args.update(sparse_mode=3, next_tokens=0, inner_precise=1)'
    if source.count(old) != 1:
        raise RuntimeError('one original causal precision argument site required')
    changed = source.replace(old, new)
    assert changed.replace(new, old) == source
    namespace = dict(function.__globals__)
    exec(compile(changed, __file__+'::causal-inner1-'+function.__name__, 'exec',
                 flags=function.__code__.co_flags & __future__.annotations.compiler_flag, dont_inherit=True), namespace)
    generated = namespace[function.__name__]
    result = FunctionType(generated.__code__, function.__globals__, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = dict(function.__annotations__)
    result.__dict__.update(function.__dict__)
    result.__module__ = function.__module__
    result.__qualname__ = function.__qualname__
    result.__doc__ = function.__doc__
    if inspect.signature(result) != inspect.signature(function):
        raise RuntimeError('complete original FIA callback signature required')
    assert result.__globals__ is function.__globals__
    return result, dict(original_source_SHA=hashlib.sha256(source.encode()).hexdigest(),
        projected_source_SHA=hashlib.sha256(changed.encode()).hexdigest(),
        exact_source_inverse=True, original_live_module_globals_preserved=True)


def install(native):
    from vllm_ascend.spec_decode.pearl import native_graph as graph, native_model as model
    old = graph.run_native_fused_infer_attention
    if model.run_native_fused_infer_attention is not old:
        raise RuntimeError('actual model FIA alias must match original graph callback')
    forward, forward_proof = project(old)
    update, update_proof = project(graph.NativeACLGraphRunner._update_fused_infer_attention_task)
    graph.run_native_fused_infer_attention = forward
    model.run_native_fused_infer_attention = forward
    graph.NativeACLGraphRunner._update_fused_infer_attention_task = update
    return dict(scope='causal inner_precise1 only; FULL ABI and all validation unchanged',
        eager_capture=forward_proof, task_update=update_proof,
        actual_model_alias_bound=True, runtime_qualification_flags_assigned=False)
