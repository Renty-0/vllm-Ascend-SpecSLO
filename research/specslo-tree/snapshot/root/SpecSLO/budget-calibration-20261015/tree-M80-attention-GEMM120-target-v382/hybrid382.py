"""Target-only Q5 geometry80; retain original120-row GEMM arithmetic."""
import ast
import copy
import inspect
import types
import __future__


def project_linear(original,linear,expected_calls=1):
    tree=ast.parse(inspect.getsource(original))
    before=ast.dump(tree,include_attributes=False)
    calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)
           and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)
           and n.func.value.id=='F'and n.func.attr=='linear']
    if len(calls)!=expected_calls:raise RuntimeError('ROOT382 exact original plain linear call count')
    originals=[copy.deepcopy(call.func)for call in calls]
    for call in calls:call.func=ast.Name(id='_ROOT382_linear',ctx=ast.Load())
    projected=copy.deepcopy(tree)
    for call,old in zip(calls,originals):call.func=old
    if ast.dump(tree,include_attributes=False)!=before:raise RuntimeError('ROOT382 complete plain function AST inverse')
    namespace=dict(original.__globals__,_ROOT382_linear=linear)
    flags=original.__code__.co_flags & __future__.annotations.compiler_flag
    exec(compile(ast.fix_missing_locations(projected),__file__+'::one-guarded-original-linear-call',
                 'exec',flags=flags,dont_inherit=True),namespace)
    result=namespace[original.__name__]
    if inspect.signature(result)!=inspect.signature(original):raise RuntimeError('ROOT382 exact original annotations and signature')
    return result


def eligible(tokens,metadata):
    return (tokens.ndim==1 and tokens.numel()==80 and metadata is not None
            and metadata.tree_attention is True
            and len(metadata.sequence_lens)==16
            and tuple(metadata.actual_seq_lengths_q)==tuple(range(5,81,5))
            and metadata.tree_attention_mask is not None
            and tuple(metadata.tree_attention_mask.shape)[:2]==(16,1)
            and metadata.tree_attention_mask.shape[2]in(10,11))


def install(engine):
    if engine.is_draft:return dict(applied=False,reason='original optimized draft retained')
    if hasattr(engine,'_hybrid382'):raise RuntimeError('ROOT382 install once before current target math')
    import torch.nn.functional as F
    from vllm_ascend.spec_decode.pearl import native_model as model
    from repair381 import install as compiler_repair
    import control380
    compiler_repair()
    state=dict(active=False,model_frames80=0,model_frames_other=0,plain_GEMM80_to120=0,
               MC2_GEMM80_to120=0,column_modules=0,head_modules=0)
    def linear(x,weight,bias=None):
        if state['active'] and x.ndim==2 and x.shape[0]==80:
            state['plain_GEMM80_to120']+=1
            return F.linear(F.pad(x,(0,0,0,40)),weight,bias)[:80]
        return F.linear(x,weight,bias)
    def mc2_linear(x,weight):
        if state['active'] and x.ndim==2 and x.shape[0]==80:
            state['MC2_GEMM80_to120']+=1
            return F.linear(F.pad(x,(0,0,0,40)),weight)[:80].contiguous()
        return F.linear(x,weight)
    old_forward=engine.model.forward
    def forward(current,input_ids,positions,attention_metadata=None):
        state['active']=eligible(input_ids,attention_metadata)
        state['model_frames80'if state['active']else'model_frames_other']+=1
        return old_forward(input_ids,positions,attention_metadata)
    engine.model.forward=types.MethodType(forward,engine.model)
    projected=project_linear(model.NativeColumnLinear.forward,linear)
    for module in engine.model.modules():
        if isinstance(module,model.NativeColumnLinear):
            if module.token_pad_multiple!=1:raise RuntimeError('ROOT382 original column padding1 required')
            module.forward=types.MethodType(projected,module);state['column_modules']+=1
    head=engine.model.lm_head
    if not isinstance(head,model.NativeLMHead)or head.token_pad_multiple!=1:
        raise RuntimeError('ROOT382 original target head and padding1 required')
    head._project_local=types.MethodType(project_linear(model.NativeLMHead._project_local,linear),head)
    state['head_modules']=1
    if state['column_modules']!=128:raise RuntimeError('ROOT382 original64 QKV and64 gate/up modules')
    model.matmul_allreduce_add_rmsnorm_or_fallback=control380.project_mc2(
        model.matmul_allreduce_add_rmsnorm_or_fallback,mc2_linear)
    engine._hybrid382=state
    return snapshot(engine)


def snapshot(engine):
    if engine.is_draft:return dict(applied=False,original_optimized_draft_unchanged=True)
    return dict(applied=True,actual=dict(engine._hybrid382),physical_target_rows=[80,120],
        attention_segments=[16,24],conditional_GEMM_rows=120,precision='BF16',quantized=False,
        original_column_padding_attributes_unchanged=True,original_qualification_bodies_unchanged=True,
        Native_qualification_flags_assigned=False,other_model_frames_take_original_linear=True,
        counters_are_Python_execution_not_graph_replay_work=True,
        no_performance_or_neural_qualification_inherited=True)
