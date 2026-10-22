"""Use unchanged original native-epilogue qualifier on live true N32 inputs."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import sys

from frame454 import extra_slots
from recorder454 import MemoryInputs
from original385 import TargetGroup

EXAMPLE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/examples/measure_specslo_rank3_reduce_add_rmsnorm.py')


class BorrowedTargetGroup(TargetGroup):
    """The original qualifier's rank0 broadcasts must stay inside target TP3."""
    def broadcast_object_list(self,objects,src=0,group=None,**kwargs):
        group=self.target if group is None else group
        if group is not self.target:raise RuntimeError('ROOT454 target-only collective')
        world_src=self.actual.get_global_rank(group,src)
        return self.actual.broadcast_object_list(objects,src=world_src,group=group,**kwargs)

    def broadcast(self,tensor,src=0,group=None,**kwargs):
        group=self.target if group is None else group
        if group is not self.target:raise RuntimeError('ROOT454 target-only broadcast')
        return self.actual.broadcast(tensor,src=self.actual.get_global_rank(group,src),group=group,**kwargs)


class LiveTensorLoader:
    """Only load is adapted: every returned tensor is observed on the real model."""
    def __init__(self,torch):self.actual=torch;self.expected_path=None;self.payload=None;self.calls=0
    def load(self,path,*,map_location,weights_only):
        if Path(path)!=self.expected_path or map_location!='cpu' or weights_only is not True or self.payload is None:
            raise RuntimeError('ROOT454 exact original payload request')
        self.calls+=1
        return {k:self.payload[k]for k in('activation','weight','residual','gamma')}
    def __getattr__(self,name):return getattr(self.actual,name)


def measure(engine,payloads,output):
    import torch
    import torch.distributed as dist
    state=engine._branch_tail_state;current=state.static_current
    if engine.is_draft or state.enabled or current['real_N']!=24 or current['real_queries']!=120:
        raise RuntimeError('ROOT454 canonical24 real B4 groups required')
    if 160 in engine.model._mc2_chained_row_counts:raise RuntimeError('ROOT454 qualify missing160, do not overwrite an existing route')
    output=Path(output)
    def save(name,value):
        with(output/f'rank{engine.rank}-{name}.json').open('w')as f:json.dump(value,f,indent=2);f.write('\n')
    def digest(value):
        cpu=value.detach().cpu().contiguous()
        return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
    parameter_pointers=[(n,v.data_ptr())for n,v in engine.model.named_parameters()]
    hashes_before={n:digest(v)for n,v in engine.model.named_parameters()}
    save('M160454-weights-before',hashes_before)
    manifest=engine.mc2_static_route_manifest;manifest_digest=manifest.digest if manifest else None
    tables=[list(row)for row in engine.cache_allocation.block_tables]
    pages=[p for row in tables for p in row if p>=0]
    if len(pages)!=len(set(pages)):raise RuntimeError('ROOT454 all32 owners have disjoint real pages')
    x,p,m,_,_=current['physical'];host_positions=p.detach().cpu().tolist()
    prefixes=engine._root445_fixture_prefixes
    owners,slots=extra_slots(host_positions,tables,[len(v)for v in prefixes])
    original_slots=m.slot_mapping.detach().cpu().tolist()
    if set(slots)&set(original_slots):raise RuntimeError('ROOT454 extra eight owners are independent')
    # Owners24..31 were separately prefilled by the unchanged72-owner bank.
    # Their prompts/roots equal owners0..7; no candidate KV is copied.
    xx=torch.cat((x,x[:40]));pp=torch.cat((p,p[:40]))
    extra_tables=torch.tensor([[v if v>=0 else state.static_zero_page for v in tables[owner]]for owner in owners],
        dtype=m.request_block_tables.dtype,device=m.request_block_tables.device)
    bt=torch.cat((m.request_block_tables,extra_tables))
    mm=replace(m,slot_mapping=torch.cat((m.slot_mapping,torch.tensor(slots,dtype=m.slot_mapping.dtype,device=m.slot_mapping.device))),
        context_lens=torch.cat((m.context_lens,m.context_lens[:40])),block_tables=bt,request_block_tables=bt,
        actual_seq_lengths_q=tuple(5*i for i in range(1,33)),sequence_lens=(2560,)*32,
        tree_attention_mask=torch.cat((m.tree_attention_mask,m.tree_attention_mask[:8])))
    if tuple(xx.shape)!=(160,)or tuple(bt.shape)!=(32,32)or tuple(mm.tree_attention_mask.shape)!=(32,1,10,4096):
        raise RuntimeError('ROOT454 exact physicalM160/S32/q5 ABI')
    if m.use_fused_infer_attention is not True or m.tree_attention is not True:raise RuntimeError('ROOT454 original FULL causal FIA')
    recorder=MemoryInputs();old=[layer.mc2_input_capture for layer in engine.model.layers]
    if any(v is not None for v in old):raise RuntimeError('ROOT454 explicit fresh diagnostic input observer')
    try:
        for layer in engine.model.layers:layer.mc2_input_capture=recorder
        with recorder.capture(dict(N=32,budgets=[4]*32)),state.active(False):
            hidden=engine.model(xx,pp,mm).clone()
        torch.npu.synchronize()
    finally:
        for layer,value in zip(engine.model.layers,old,strict=True):layer.mc2_input_capture=value
    captured=recorder.payloads()
    if not bool(torch.isfinite(hidden).all()):raise RuntimeError('ROOT454 real32 model frame finite')
    descriptor=dict(physical_M=160,physical_S=32,real_requests=32,real_query_rows=160,
        owner_ids=list(range(32)),additional_owner_ids=owners,
        additional_owners_separately_prefilled=True,known_equal_triplet_roots_from_original_bank=True,
        original24_slots=original_slots,additional40_slots=slots,
        all_real_slots_disjoint=len(set(original_slots+slots))==160,
        ordinary_prompt_prefill_unchanged=True,new160_profile_installed=False,Native_flags_assigned=False,
        actual_hidden_sha256=digest(hidden),captured=recorder.summary(),tensors_serialized=False,
        parameter_weights_readonly=True,full463=False,Raw=None,Goodput=None)
    if not descriptor['all_real_slots_disjoint']:raise RuntimeError('ROOT454 all160 query writes unique')
    save('M160454-real32-capture',descriptor)
    name='_ROOT454_unchanged_native_epilogue_qualifier'
    spec=importlib.util.spec_from_file_location(name,EXAMPLE);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    group=engine.model.layers[0].context.group;proxy=BorrowedTargetGroup(dist,group);module.dist=proxy
    loader=LiveTensorLoader(torch);module.torch=loader
    runtime=module.initialize_measurement_runtime(chained_flush=True,require_production_operator=True)
    if runtime.rank!=proxy.get_rank()or runtime.local_rank!=engine.rank:raise RuntimeError('ROOT454 exact TP rank and actual local device')
    if hasattr(engine,'_root454_measurement_resources'):raise RuntimeError('ROOT454 one retained campaign')
    engine._root454_measurement_resources=(module,runtime,proxy,loader,recorder)
    results=[]
    for(layer,kind),value in sorted(captured.items()):
        k=3072 if kind=='attention'else 8576
        folder=output/'live-M160454'/kind/f'layer-{layer}'/'m160';folder.mkdir(parents=True,exist_ok=True)
        loader.expected_path=folder/f'rank-{runtime.rank}.pt';loader.payload=value
        args=SimpleNamespace(m=160,k=k,n=5120,input_dir=folder,warmup_replays=40,replays_per_sample=50,samples=31,
            changed_input_replays=1024,changed_input_delta=0.015625,execution_mode='graph',operations_per_graph=1,
            chained_flush=True,output=folder/'measurement.json',production_profile_output=folder/'single-shape-qualified-profile.json',
            latency_percentile=95.,minimum_speedup=1.02)
        before_inputs={field:digest(value[field])for field in('activation','residual')}
        result=module.run_measurement_job(args,runtime)
        recorder.validate_readonly_references()
        row=dict(layer=layer,projection=kind,M=160,K=k,N=5120,measurement=str(args.output),
            original_qualifier_admission_rejection=result,production_row_accepted=result is None,
            original_input_hashes=before_inputs,model_weight_pointer=value['weight_pointer'],gamma_pointer=value['gamma_pointer'],
            payload_loader='in-memory real32 observed tensors, no .pt written',changed_input_replays=1024,
            original_numerical_and_p95_functions_unchanged=True)
        results.append(row);save('M160454-qualification-progress',results)
    module.finalize_measurement_runtime(runtime)
    if loader.calls!=6 or proxy.borrowed_destroy_calls!=1:raise RuntimeError('ROOT454 six real loads and borrowed communicator lifecycle')
    hashes_after={n:digest(v)for n,v in engine.model.named_parameters()};save('M160454-weights-after',hashes_after)
    if hashes_before!=hashes_after or parameter_pointers!=[(n,v.data_ptr())for n,v in engine.model.named_parameters()]:
        raise RuntimeError('ROOT454 model parameter bytes and storage unchanged')
    if(manifest.digest if manifest else None)!=manifest_digest:raise RuntimeError('ROOT454 old MC2 profile and manifest unchanged')
    local=dict(rank=engine.rank,TP_rank=runtime.rank,real32_capture=descriptor,results=results,
        original_qualifier=str(EXAMPLE),original_qualifier_sha256=hashlib.sha256(EXAMPLE.read_bytes()).hexdigest(),
        borrowed_TP3_group=True,actual_init_noops=proxy.borrowed_init_checks,actual_destroy_noops=proxy.borrowed_destroy_calls,
        payload_loader_calls=loader.calls,model_weights_serialized=False,model_parameters_SHA_unchanged=True,
        old_profile_and_manifest_unchanged=True,new_profile_installed=False,Native_flags_assigned=False,
        graph_and_input_resources_held_until_owned_worker_teardown=True,production_permission=False,full463=False,Raw=None,Goodput=None)
    all3=[None]*3;dist.all_gather_object(all3,local,group=group)
    report=dict(schema='ROOT454.real32-live-M160-native-epilogue-qualification.v1',rank=engine.rank,by_target_rank=all3,
        all6_original_operator_gates_accepted=all(v['production_row_accepted']for v in results),
        production_permission=False,full463=False,Raw=None,Goodput=None,adopted=False)
    path=output/f'rank{engine.rank}-M160454-final.json'
    with path.open('x')as f:json.dump(report,f,indent=2);f.write('\n')
    return dict(report=str(path),all_parameter_storages_unchanged=True,full463=False,Native_flags_assigned=False,production_permission=False,Raw=None,Goodput=None)
