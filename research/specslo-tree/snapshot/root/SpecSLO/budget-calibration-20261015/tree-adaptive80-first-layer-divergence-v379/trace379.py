"""Observe real eager intermediates; hooks return None and change no values."""
import hashlib
import json


def aligned_shape(baseline,candidate):
    if baseline==candidate:return None
    if len(baseline)!=len(candidate):raise ValueError('ROOT379 same tensor rank required')
    axes=[i for i,(a,b)in enumerate(zip(baseline,candidate))if a!=b]
    if len(axes)!=1 or baseline[axes[0]]*80!=candidate[axes[0]]*120:
        raise ValueError('ROOT379 only original120/candidate80 batch axis may differ')
    return axes[0]


class Trace:
    def __init__(self,engine):
        self.engine=engine;self.lane=None;self.values={};self.order={};self.handles=[];self.bytes=0
        self.norm_type=type(engine.model.layers[0].input_layernorm)
        self.old_mc2=self.norm_type.forward_mc2
        owner=self
        def observed_mc2(norm,*args,**kwargs):
            route=kwargs.get('static_route')
            selected=route is not None and int(route.layer_index)==0
            label='MC2.layer0/'+str(route.projection_kind) if selected else ''
            if selected:owner.capture(label+'/input',args[:2])
            result=owner.old_mc2(norm,*args,**kwargs)
            if selected:owner.capture(label+'/output',result)
            return result
        self.norm_type.forward_mc2=observed_mc2
        targets=[('embedding',engine.model.embed_tokens)]
        for i,layer in enumerate(engine.model.layers):
            if i==0:
                targets.extend(('layers.0'+('.'+name if name else ''),module)
                               for name,module in layer.named_modules())
            else:
                self.handles.append(layer.register_forward_hook(self.post(f'layers.{i}')))
        if hasattr(engine.model,'norm'):targets.append(('final_norm',engine.model.norm))
        for name,module in targets:
            self.handles.append(module.register_forward_pre_hook(self.pre(name)))
            self.handles.append(module.register_forward_hook(self.post(name)))

    def capture(self,label,value):
        if self.lane is None:return
        import torch
        if isinstance(value,torch.Tensor):
            # This is a diagnostic of completed values, not a latency test.
            # The profile is explicitly checked to flush each MC2 route.
            torch.npu.synchronize()
            saved=value.detach().clone()
            self.bytes+=saved.numel()*saved.element_size()
            if self.bytes>768*1024**2:raise RuntimeError('ROOT379 bounded transient NPU trace memory')
            lane=self.values.setdefault(self.lane,{})
            key=label
            suffix=0
            while key in lane:
                suffix+=1;key=label+f'#{suffix}'
            lane[key]=saved;self.order.setdefault(self.lane,[]).append(key)
        elif isinstance(value,(tuple,list)):
            for i,item in enumerate(value):self.capture(label+f'/{i}',item)
        elif isinstance(value,dict):
            for key in sorted(value):self.capture(label+'/'+str(key),value[key])

    def pre(self,name):
        def hook(module,args):self.capture(name+'/input',args)
        return hook

    def post(self,name):
        def hook(module,args,output):self.capture(name+'/output',output)
        return hook

    def compare(self):
        import torch
        baseline=self.values['fixed120'];candidate=self.values['adaptive80']
        rows=[]
        for name in self.order['fixed120']:
            if name not in candidate:
                rows.append(dict(name=name,missing_in='adaptive80'));continue
            a,b=baseline[name],candidate[name]
            try:axis=aligned_shape(tuple(a.shape),tuple(b.shape))
            except ValueError:
                rows.append(dict(name=name,incompatible_shapes=[list(a.shape),list(b.shape)]));continue
            original_shape=list(a.shape)
            if axis is not None:
                slices=[slice(None)]*a.ndim;slices[axis]=slice(0,b.shape[axis]);a=a[tuple(slices)]
            equal=torch.equal(a,b)
            af=a.float() if not a.is_complex() else a
            bf=b.float() if not b.is_complex() else b
            absolute=(af-bf).abs().float()
            changed=(a!=b)
            count=int(changed.count_nonzero().item())
            first=None
            if count:
                first=changed.nonzero()[0].cpu().tolist()
            rows.append(dict(name=name,baseline_full_shape=original_shape,candidate_shape=list(b.shape),
                aligned_shape=list(a.shape),cropped_batch_axis=axis,dtype=str(a.dtype),
                bitwise_equal=equal,different_elements=count,elements=a.numel(),
                max_abs=float(absolute.max().item())if a.numel() else 0.0,
                first_difference_index=first))
        for name in self.order['adaptive80']:
            if name not in baseline:rows.append(dict(name=name,missing_in='fixed120'))
        first=next((r for r in rows if r.get('bitwise_equal')is False),None)
        layer_outputs=[r for r in rows if r['name'].split('/')[0].startswith('layers.')
                       and r['name'].split('/')[0][len('layers.'):].isdigit()and '/output' in r['name']]
        return dict(rows=rows,first_common_tensor_difference=first,
            first_decoder_output_difference=next((r for r in layer_outputs if r.get('bitwise_equal')is False),None),
            transient_trace_bytes=self.bytes,raw_tensors_saved_to_disk=False,
            hooks_change_values=False,diagnostic_synchronizes_before_capture=True,
            original_mc2_impl_called_once_per_route=True,performance_measured=False,model_qualified=False)

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles.clear()
        self.norm_type.forward_mc2=self.old_mc2
