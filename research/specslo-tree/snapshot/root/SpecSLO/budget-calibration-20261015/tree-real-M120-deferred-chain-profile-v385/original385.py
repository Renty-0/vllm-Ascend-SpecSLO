"""Run the original chain diagnostic on the borrowed real TP3 group and live payloads."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

EXAMPLE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/examples/check_specslo_mc2_deferred_chain.py')


class TargetGroup:
    def __init__(self,dist,group):
        self.actual=dist;self.target=group;self.group=SimpleNamespace(WORLD=group)
        self.borrowed_init_checks=0;self.borrowed_destroy_calls=0

    def get_world_size(self,group=None):return self.actual.get_world_size(group=self.target if group is None else group)
    def get_rank(self,group=None):return self.actual.get_rank(group=self.target if group is None else group)
    def init_process_group(self,*args,**kwargs):
        if not self.actual.is_initialized()or self.get_world_size()!=3:
            raise RuntimeError('ROOT385 original existing TP3 group required')
        self.borrowed_init_checks+=1

    def destroy_process_group(self,*args,**kwargs):
        # The outer original four-role worker owns this group and its cleanup.
        self.borrowed_destroy_calls+=1

    def all_reduce(self,*args,**kwargs):
        kwargs.setdefault('group',self.target);return self.actual.all_reduce(*args,**kwargs)
    def all_gather_object(self,*args,**kwargs):
        kwargs.setdefault('group',self.target);return self.actual.all_gather_object(*args,**kwargs)
    def barrier(self,*args,**kwargs):
        kwargs.setdefault('group',self.target);return self.actual.barrier(*args,**kwargs)
    def __getattr__(self,name):return getattr(self.actual,name)


def run(engine,payloads,output):
    import torch.distributed as dist
    spec=importlib.util.spec_from_file_location('_ROOT385_original_deferred_chain',EXAMPLE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    group=engine.model.layers[0].context.group
    proxy=TargetGroup(dist,group);module.dist=proxy
    root=Path(output)/'same-instance-live-payloads385'
    dirs={name:root/name for name in ['attention','down']}
    for name,folder in dirs.items():
        folder.mkdir(parents=True,exist_ok=True)
        value=payloads[name]
        descriptor=dict(rank=engine.rank,TP_rank=proxy.get_rank(),layer=0,projection=name,
            inputs='actual captured NativeRMSNorm.forward_mc2 arguments from current M120 model frame',
            tensor_shapes={k:list(t.shape)for k,t in value.items()},dtype='BF16',
            tensors_serialized=False,weights_reference='current immutable model parameter',
            original_example_payload_loader_adapted_to_live_tensors=True)
        (folder/f'rank{engine.rank}-descriptor.json').write_text(json.dumps(descriptor,indent=2)+'\n')
    args=SimpleNamespace(attention_input_dir=dirs['attention'],down_input_dir=dirs['down'],
        execution_mode='graph',layer_count=64,warmup_replays=20,replays_per_sample=20,
        samples=9,changed_input_replays=5,changed_input_delta=0.015625,
        output=Path(output)/'actual-original-M120-deferred-chain385.json')
    module._parser=lambda:SimpleNamespace(parse_args=lambda:args)
    def load(path,rank):
        if rank!=proxy.get_rank():raise RuntimeError('ROOT385 exact actual TP3 rank')
        name=next(name for name,folder in dirs.items()if folder==path)
        return dict(payloads[name])
    module._load_payload=load
    # Original main, mathematical comparisons, graph capture, changed inputs,
    # sample ordering, measurements and report construction remain unchanged.
    module.main()
    return dict(original_example=str(EXAMPLE),original_main_unchanged=True,
        original_numerical_and_measurement_functions_unchanged=True,
        borrowed_existing_TP3_group=True,TP_world_size=proxy.get_world_size(),TP_rank=proxy.get_rank(),
        actual_init_checks=proxy.borrowed_init_checks,actual_destroy_noops=proxy.borrowed_destroy_calls,
        actual_layers_repeated=64,actual_epilogues_per_chain=128,live_payloads_not_serialized=True,
        report=str(args.output),performance_scope='operator sequence only, no full generation',
        Native_flags_assigned=False,Raw=None,Goodput=None)
