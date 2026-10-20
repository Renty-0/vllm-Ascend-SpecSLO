"""Original all4 completion fence, before any following model/TP call."""
import time
from scope151 import require

def stage4(e,events,label,callback):
 import torch
 import torch.distributed as dist
 value=None;error=None;sequence=len(events);start=time.perf_counter()
 try:value=callback();torch.npu.synchronize()
 except BaseException as exc:error=type(exc).__name__+': '+str(exc)
 rows=[None]*4
 dist.all_gather_object(rows,dict(rank=e.rank,sequence=sequence,stage=label,error=error),group=e.groups.verification_coordination_group)
 events.append(dict(sequence=sequence,stage=label,ranks=rows,callback_begin=start,all4_end=time.perf_counter(),cost_sample=False))
 require(sorted(r['rank'] for r in rows)==list(range(4)) and all(r['sequence']==sequence and r['stage']==label and r['error'] is None for r in rows),
  'common all4 boundary rejected '+label+': '+repr(rows))
 return value
