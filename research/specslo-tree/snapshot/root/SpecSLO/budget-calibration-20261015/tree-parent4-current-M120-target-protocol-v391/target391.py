"""Original199 full-history target proof with an independently rebuilt M120 ABI."""
from types import FunctionType
import target199 as original
from metadata390 import physical


def target_shape(budgets):
    from mapper255 import envelope
    values=tuple(budgets)
    if not(1<=len(values)<=24 and all(type(x)is int and x==4 for x in values)):
        raise RuntimeError('ROOT391 finite allB4 target scope')
    m,s,padding=envelope(len(values),5*len(values),'causal')
    return dict(budgets=list(values),N=len(values),C=sum(values),real_Q=5*len(values),
        M=m,S=s,scratch=padding,family='static_FULL_M120_Q5_S24')


def cloned():
    namespace=dict(vars(original),metadata=physical,target_shape=target_shape)
    names=('padded','clone_output','original_call','own_compact','rebuild_history','qualify')
    for name in names:
        function=getattr(original,name)
        namespace[name]=FunctionType(function.__code__,namespace,function.__name__,
            function.__defaults__,function.__closure__)
        namespace[name].__kwdefaults__=function.__kwdefaults__
    base=namespace['original_call']
    def observed(e,frame,*,changed=False):
        import torch
        result=base(e,frame,changed=changed)
        expected=physical(e,frame,changed=changed)
        current=e._branch_tail_state.static_current
        if current is None or current['copy_enabled']:
            raise RuntimeError('ROOT391 actual finite allB4 static dispatch required')
        x,pos,meta,_,mapping=current['physical']
        xx,pp,mm=expected
        tensor_fields=('slot_mapping','context_lens','block_tables','request_block_tables','tree_attention_mask')
        scalars=('actual_seq_lengths_q','sequence_lens','tree_attention','use_fused_infer_attention')
        if not(torch.equal(x,xx)and torch.equal(pos,pp)
                and all(torch.equal(getattr(meta,k),getattr(mm,k))for k in tensor_fields)
                and all(getattr(meta,k)==getattr(mm,k)for k in scalars)
                and mapping['physical_to_original']==list(range(120))):
            raise RuntimeError('ROOT391 all actual Native physical metadata differs from independent builder')
        binding=e._parent4_target391_binding
        binding['actual_metadata_matches']=binding.get('actual_metadata_matches',0)+1
        return result
    namespace['original_call']=observed
    return namespace


_namespace=cloned()
for _name in ('padded','clone_output','original_call','own_compact','rebuild_history','qualify'):
    globals()[_name]=_namespace[_name]
