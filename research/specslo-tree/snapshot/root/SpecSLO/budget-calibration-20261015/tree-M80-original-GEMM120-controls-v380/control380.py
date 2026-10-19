"""Temporary eager-only GEMM shape controls; preserve precision and dispatch gates."""
import ast
import copy
import inspect


def project_mc2(original,linear):
    tree=ast.parse(inspect.getsource(original))
    candidates=[n for n in ast.walk(tree)if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
        and n.targets[0].id=='local_projection']
    if len(candidates)!=1:raise RuntimeError('ROOT380 one original native epilogue local projection')
    call=candidates[0].value
    if ast.unparse(call)!='F.linear(x, weight)':raise RuntimeError('ROOT380 exact original local projection')
    before=ast.dump(tree,include_attributes=False)
    old=copy.deepcopy(call.func)
    call.func=ast.Name(id='_ROOT380_shape_control_linear',ctx=ast.Load())
    projected=copy.deepcopy(tree)
    call.func=old
    if ast.dump(tree,include_attributes=False)!=before:raise RuntimeError('ROOT380 complete MC2 function AST inverse')
    namespace=dict(original.__globals__,_ROOT380_shape_control_linear=linear)
    exec(compile(ast.fix_missing_locations(projected),__file__+'::original-MC2-one-local-GEMM-shape-control','exec'),namespace)
    result=namespace[original.__name__]
    if inspect.signature(result)!=inspect.signature(original):raise RuntimeError('ROOT380 original MC2 signature')
    return result


class Control:
    def __init__(self,engine,mode):
        if mode not in ('none','qkv120','columns120','columns_and_mc2_120'):
            raise ValueError('ROOT380 explicit bounded control')
        self.engine=engine;self.mode=mode;self.saved=[];self.old_mc2=None
        self.counts=dict(mode=mode,padded_column_modules=0,MC2_projection_calls=0,
            MC2_local_projection_rows=[80,120],qualification_bodies_changed=False,
            qualification_flags_assigned=False,weights_or_precision_changed=False)

    def __enter__(self):
        if self.mode=='none':return self
        from vllm_ascend.spec_decode.pearl import native_model as model
        for name,module in self.engine.model.named_modules():
            selected=(name.endswith('self_attn.qkv_proj') if self.mode=='qkv120' else
                isinstance(module,(model.NativeColumnLinear,model.NativeLMHead)))
            if selected:
                old=module.token_pad_multiple
                if old!=1:raise RuntimeError('ROOT380 original production GEMM row padding expected1')
                self.saved.append((module,old));module.token_pad_multiple=120
                self.counts['padded_column_modules']+=1
        if self.counts['padded_column_modules']!=(64 if self.mode=='qkv120' else 129):
            raise RuntimeError('ROOT380 actual target64 QKV plus64 gate/up plus1 head modules')
        if self.mode=='columns_and_mc2_120':
            import torch.nn.functional as F
            self.old_mc2=model.matmul_allreduce_add_rmsnorm_or_fallback
            def linear(x,weight):
                if x.ndim==2 and x.shape[0]==80:
                    self.counts['MC2_projection_calls']+=1
                    return F.linear(F.pad(x,(0,0,0,40)),weight)[:80].contiguous()
                return F.linear(x,weight)
            model.matmul_allreduce_add_rmsnorm_or_fallback=project_mc2(self.old_mc2,linear)
        return self

    def __exit__(self,*args):
        if self.old_mc2 is not None:
            from vllm_ascend.spec_decode.pearl import native_model as model
            model.matmul_allreduce_add_rmsnorm_or_fallback=self.old_mc2
        for module,value in self.saved:module.token_pad_multiple=value
        self.counts['original_column_padding_restored']=all(module.token_pad_multiple==value for module,value in self.saved)
        self.counts['original_mc2_dispatch_restored']=self.old_mc2 is None or model.matmul_allreduce_add_rmsnorm_or_fallback is self.old_mc2
        return False
