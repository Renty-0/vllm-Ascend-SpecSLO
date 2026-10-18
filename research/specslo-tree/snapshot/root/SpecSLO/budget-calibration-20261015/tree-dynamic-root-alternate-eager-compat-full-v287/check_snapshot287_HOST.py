"""Bounded owned snapshot writes and no algorithm state mutation."""
import json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
P=Path(__file__).resolve().parent
sys.path.append(str(P.parent/'tree-paired-alternate-gain-host-v274'))
from debug_snapshot287 import maybe_save
with tempfile.TemporaryDirectory(prefix='ROOT287-snapshot-',dir=P)as folder:
 policy=SimpleNamespace(_ROOT274_pairs=dict(generation=5,by_generation={'5':{}},original_gain=object()),counts={'old_counter':7})
 e=SimpleNamespace(rank=0,_ROOT287_debug_output=folder,_ROOT287_debug_observations=255,_dynamic_tree_generation=5,_ROOT287_policy=policy,_ROOT287_draft_counts={'5':{'calls':3}},_ROOT287_request_gain_observations={'5':{'17':{'samples':1}}})
 request=SimpleNamespace(request_index=17);plan=SimpleNamespace(candidate_budget=8)
 maybe_save(e,request,plan,[-1]*4,[19,-1,-1,-1,-1])
 path=Path(folder)/'rank0-online-state-latest.json';raw=path.read_bytes();v=json.loads(raw)
 assert v['generation']==5 and v['last_verdict']['request']==17 and v['paired_gain']=={'generation':5,'by_generation':{'5':{}}}
 assert v['observations_since_process_start']==256 and policy.counts=={'old_counter':7}
 maybe_save(e,request,plan,[-1]*4,[20,-1,-1,-1,-1]);assert path.read_bytes()==raw
 assert not path.with_suffix('.tmp').exists()
 result=dict(status='HOST_periodic_bounded_snapshot_and_state_preservation_passed',no_tensor_or_device_operations=True,actual_write_bytes=len(raw),every256_rows=True,atomic_replace=True,original_policy_counts_unchanged=True)
with(P/'HOST_SNAPSHOT287_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
