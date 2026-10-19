"""HOST prototype: actual165 projection with original fully flushed chain ABI.

FIA remains160/S32. No profile160 entry or deferred-read-done admission.
"""
import ast
import copy
import inspect
from math import prod
import textwrap
from types import FunctionType, MethodType

import bridge330 as parent


def projected_rows(hidden, metadata):
    rows=prod(hidden.shape[:-1])
    if (rows==160 and metadata is not None and metadata.tree_attention
            and tuple(metadata.actual_seq_lengths_q)==tuple(range(5,161,5))):
        return 165
    return rows


def projected_forward(function):
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    original=ast.dump(tree,include_attributes=False)
    nodes=[n for n in ast.walk(tree)if isinstance(n,ast.Assign)
           and any(isinstance(t,ast.Name)and t.id=='row_count'for t in n.targets)]
    if len(nodes)!=1 or ast.unparse(nodes[0].value)!='prod(hidden_states.shape[:-1])':
        raise RuntimeError('One exact original model chain admission row assignment required')
    prior=nodes[0].value
    nodes[0].value=ast.parse('_actual_projection_rows334(hidden_states, attention_metadata)',mode='eval').body
    compiled=copy.deepcopy(tree)
    nodes[0].value=prior
    if ast.dump(tree,include_attributes=False)!=original:
        raise RuntimeError('Whole original NativeModel row-admission AST inverse')
    namespace=dict(function.__globals__,_actual_projection_rows334=projected_rows)
    import __future__
    flags=function.__code__.co_flags & __future__.annotations.compiler_flag
    exec(compile(ast.fix_missing_locations(compiled),__file__+'::actual165-projection-chain-admission','exec',flags=flags,dont_inherit=True),namespace)
    return namespace[function.__name__]


def adapted_method(function, replacements):
    source=textwrap.dedent(inspect.getsource(function))
    inverse=source
    for before,after in replacements:
        if source.count(before)!=1:raise RuntimeError('Exact334 bridge source anchor '+before)
        source=source.replace(before,after)
    restored=source
    for before,after in reversed(replacements):restored=restored.replace(after,before)
    if restored!=inverse:raise RuntimeError('Whole parent330 bridge source inverse')
    namespace=dict(function.__globals__)
    exec(compile(source,__file__+'::original330-adapter-chain-binding','exec'),namespace)
    return namespace[function.__name__]


class FlushedProjectionBridge(parent.ProjectionBridge):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.active_chain=None

    forward=adapted_method(parent.ProjectionBridge.forward,[
        ('or kwargs.get("mc2_chain") is not None','or kwargs.get("mc2_chain") is not self.active_chain')])
    scope=adapted_method(parent.ProjectionBridge.scope,[
        ('if kwargs.get("mc2_chain") is not None:', 'if kwargs.get("mc2_chain") is None:'),
        ('raise RuntimeError("Physical160 has no qualified deferred MC2 chain")', 'raise RuntimeError("Physical160 projection165 requires original qualified flushed chain")'),
        ('self.active_layer = index','self.active_layer = index\n    self.active_chain = kwargs["mc2_chain"]\n    if self.active_chain.defer_read_done is not False:\n        raise RuntimeError("No deferred-read-done evidence for this new model geometry")'),
        ('self.active_layer = None','self.active_layer = None\n        self.active_chain = None')])

    def summary(self):
        result=super().summary()
        result.update(original_165_standalone_chained_flush_ABI=True,
                      deferred_chain_enabled=False,chain_scope_closed=self.active_chain is None)
        return result


def install(model,native_model):
    if 165 not in model._mc2_chained_row_counts or 165 in model._mc2_deferred_row_counts:
        raise RuntimeError('Require original165 flushed qualification and reject deferred admission')
    original=model.forward.__func__
    forward=projected_forward(original)
    install_function=FunctionType(parent.install.__code__,
        dict(parent.install.__globals__,ProjectionBridge=FlushedProjectionBridge),
        parent.install.__name__,parent.install.__defaults__,parent.install.__closure__)
    bridge=install_function(model,native_model)
    model.forward=MethodType(forward,model)
    model._specslo_bridge334=bridge
    return bridge
