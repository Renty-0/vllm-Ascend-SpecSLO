"""Fail closed on actual resident/cache memory; no allocator/environment edits."""
import json
import time
from pathlib import Path

GIB = 1024**3
# 64GiB physical devices: preserve 4GiB outside the torch allocator for the
# driver and 6GiB for subsequent qualification/service transient allocation.
# mem_get_info is additionally authoritative about actual remaining memory.
MAX_TORCH_RESERVED = 54*GIB
MAX_TORCH_PEAK_ALLOCATED = 54*GIB
MIN_DEVICE_FREE = 6*GIB


def snapshot(engine):
    import torch
    from prefill_arena import retained_task_storage
    from workspace_storage import snapshot as workspace_storage
    free,total = torch.npu.mem_get_info(engine.device)
    runner = engine.graph_runner
    return dict(npu_allocated_bytes=int(torch.npu.memory_allocated(engine.device)),
        npu_reserved_bytes=int(torch.npu.memory_reserved(engine.device)),
        npu_peak_allocated_bytes=int(torch.npu.max_memory_allocated(engine.device)),
        npu_free_bytes=int(free), npu_total_bytes=int(total),
        graph_entries=sum(len(getattr(runner,name))
                          for name in ('entries','draft_entries','target_entries')),
        **retained_task_storage(runner),**workspace_storage(runner))


def reasons(values):
    result=[]
    if values['npu_reserved_bytes'] > MAX_TORCH_RESERVED: result.append('reserved_headroom')
    if values['npu_peak_allocated_bytes'] > MAX_TORCH_PEAK_ALLOCATED: result.append('peak_headroom')
    if values['npu_free_bytes'] < MIN_DEVICE_FREE: result.append('driver_free_headroom')
    if values['graph_entries'] > 512: result.append('native_entry_capacity')
    return result


def checkpoint(engine, phase, completed, total, *, detail=None):
    """All4 ranks call in the same order, including their idle-role ranks."""
    import torch
    import torch.distributed as dist
    torch.npu.synchronize()
    values=snapshot(engine)
    errors=reasons(values)
    if engine.graph_runner.max_graph_entries!=512:
        errors.append('changed_native_entry_limit')
    if bool(engine._spec_rhythm_nonfinite_flag(include_layer_cache=False).cpu().item()):
        errors.append('native_sticky_nonfinite')
    record=dict(phase=phase,completed=completed,total=total,rank=int(engine.rank),
                updated_unix=time.time(),memory=values,local_errors=errors,detail=detail)
    path=Path(__file__).resolve().parent/'startup-progress'
    path.mkdir(exist_ok=True)
    with (path/f'finite-inventory-rank{engine.rank}.jsonl').open('a') as out:
        out.write(json.dumps(record)+'\n')
    vote=torch.tensor([int(bool(errors))],dtype=torch.int32)
    dist.all_reduce(vote,op=dist.ReduceOp.MAX,group=engine.groups.verification_coordination_group)
    if int(vote.item()):
        raise RuntimeError('Finite inventory failed memory/finite gate on at least one rank: '+str(errors))
    return record
