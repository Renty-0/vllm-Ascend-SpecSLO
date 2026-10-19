"""Original value-preserving trace plus layer0 RoPE/cache and FIA boundaries."""
import types
from trace379 import Trace as Original


class Trace(Original):
    def __init__(self,engine):
        super().__init__(engine)
        self.attn=engine.model.layers[0].self_attn
        self.method_saves=[]
        for name in ('_write_to_cache','_fused_infer_attention'):
            old=getattr(self.attn,name)
            own=self.attn.__dict__.get(name)
            self.method_saves.append((name,own))
            def wrapped(current,*args,_name=name,_old=old,**kwargs):
                self.capture('layers.0.self_attn.'+_name+'/input',args)
                result=_old(*args,**kwargs)
                self.capture('layers.0.self_attn.'+_name+'/output',result)
                return result
            setattr(self.attn,name,types.MethodType(wrapped,self.attn))

    def forget_candidate(self):
        self.values.pop('adaptive80',None);self.order.pop('adaptive80',None)
        self.bytes=sum(v.numel()*v.element_size()for rows in self.values.values()for v in rows.values())

    def close(self):
        for name,own in self.method_saves:
            if own is None:delattr(self.attn,name)
            else:setattr(self.attn,name,own)
        super().close()
