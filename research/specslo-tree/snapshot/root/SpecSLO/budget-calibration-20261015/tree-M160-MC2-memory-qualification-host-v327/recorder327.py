"""Bounded actual-model MC2 inputs; keep weights in the original model."""
from contextlib import contextmanager
from types import MappingProxyType


class MemoryInputs:
    projection_kind='attention'

    def __init__(self):
        self.active=False
        self.sealed=False
        self._payloads={}

    @contextmanager
    def capture(self,case):
        if self.active or self.sealed:
            raise RuntimeError('capture is a single eager N32/B4 invocation')
        if case['N']!=32 or case['budgets']!=[4]*32:
            raise ValueError('all160 captured query rows must be real N32/B4')
        self.active=True
        try:
            yield self
            if set(self._payloads)!={(layer,kind)for layer in (0,31,63)for kind in ('attention','down')}:
                raise RuntimeError('all six real layer/projection payloads required')
            self.sealed=True
        finally:
            self.active=False

    def maybe_capture(self,activation,weight,residual,gamma,*,projection_kind,layer_index):
        if not self.active or layer_index not in (0,31,63):
            return
        if projection_kind not in ('attention','down'):
            raise ValueError('explicit attention/down route')
        import torch
        k=3072 if projection_kind=='attention'else 8576
        tensors=(activation,weight,residual,gamma)
        shapes=((160,k),(5120,k),(160,5120),(5120,))
        if any(tuple(v.shape)!=shape for v,shape in zip(tensors,shapes,strict=True)):
            raise ValueError('actual dense M160/TP3 projection shapes required')
        if any(v.device!=activation.device or v.dtype!=torch.bfloat16 or not v.is_contiguous()for v in tensors):
            raise ValueError('same-device contiguous BF16 actual inputs')
        if activation.device.type=='npu'and torch.npu.is_current_stream_capturing():
            raise RuntimeError('read actual eager inputs outside ACLGraph capture')
        key=layer_index,projection_kind
        if key in self._payloads:
            raise RuntimeError('one observation per actual layer/projection')
        self._payloads[key]=dict(activation=activation.detach().clone(),weight=weight.detach(),
            residual=residual.detach().clone(),gamma=gamma.detach(),
            weight_pointer=weight.data_ptr(),gamma_pointer=gamma.data_ptr(),
            original_weight=weight,original_gamma=gamma)

    def payloads(self):
        if not self.sealed:
            raise RuntimeError('complete eager capture required')
        self.validate_readonly_references()
        return MappingProxyType(self._payloads)

    def validate_readonly_references(self):
        for payload in self._payloads.values():
            if not(payload['weight'].data_ptr()==payload['original_weight'].data_ptr()==payload['weight_pointer']):
                raise RuntimeError('model weight pointer changed')
            if not(payload['gamma'].data_ptr()==payload['original_gamma'].data_ptr()==payload['gamma_pointer']):
                raise RuntimeError('model normalization pointer changed')
            if payload['weight'].data_ptr()!=payload['weight_pointer']or payload['gamma'].data_ptr()!=payload['gamma_pointer']:
                raise RuntimeError('original model parameter reference changed')

    def summary(self):
        return dict(sealed=self.sealed,entries=[dict(layer=layer,projection=kind,
            shape=list(payload['activation'].shape),extra_cloned_bytes=(payload['activation'].numel()+payload['residual'].numel())*2,
            original_model_weight_reference=True,original_model_gamma_reference=True)
            for(layer,kind),payload in sorted(self._payloads.items())],
            model_weights_serialized=False,payloads_serialized=False,model_qualified=False,Goodput=None)
