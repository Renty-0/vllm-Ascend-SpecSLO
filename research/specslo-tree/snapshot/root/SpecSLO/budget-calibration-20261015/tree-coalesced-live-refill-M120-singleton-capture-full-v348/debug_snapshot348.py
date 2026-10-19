"""Bounded diagnostic-only snapshots; no tensor or device operations."""
import json,time
from pathlib import Path

def maybe_save(engine,request,plan,accepted,outputs):
    output=getattr(engine,'_ROOT348_debug_output',None)
    if output is None or engine.rank!=0:return
    engine._ROOT348_debug_observations=getattr(engine,'_ROOT348_debug_observations',0)+1
    count=engine._ROOT348_debug_observations
    if count%256:return
    from risk348 import snapshot
    generation=str(engine._dynamic_tree_generation)
    value=dict(generation=int(generation),observations_since_process_start=count,
        monotonic=time.monotonic(),performance_qualified=False,
        last_verdict=dict(request=request.request_index,B=plan.candidate_budget,
            accepted=list(accepted),outputs=list(outputs)),
        paired_gain=snapshot(engine._ROOT348_policy),
        draft_counts=engine._ROOT348_draft_counts,
        request_gain_observations=engine._ROOT348_request_gain_observations,
        policy_counts=dict(engine._ROOT348_policy.counts))
    path=Path(output)/'rank0-online-state-latest.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,separators=(',',':')));temp.replace(path)
