"""One R1 discovery scope; no full qualification or performance claim."""
import hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
P151=B/'tree-normal4-parent3-shared4-eager-v151'
P152=B/'tree-union-shared4-protocol-local-v152'
for index,path in enumerate((P152,P151),1):
 if str(path) not in sys.path:sys.path.insert(index,str(path))
from scope152 import Case,bucket,target_shape,make_config,NATIVE_COUNTERS,FAILURES
ACTION='natural-positive-union-discovery-local'
def require(value,message):
 if value is not True:raise RuntimeError('V154: '+message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def case():return Case(1)
def contract():
 return dict(repair_count=1,repair_owners=[0],normal=32,eager=32,real_prefix=128,parent_B=3,child_B=3,
  normal_C=96,child_C=96,current_target_C=96,current_target_Q=128,parent_target_M=128,
  union_M=36,frontier_head_M=32,shared_chain_M=64,shared_steps=4,
  eager_real_prefix_candidates=16,eager_old_negative_prefixes=16,
  old_positive_ticket_link_retained=False,new_natural_positive_not_assumed=True,
  no_target_token_injection=True,zero_positive_early_stop=True,all_positive_is_discovery_success=True,
  two_complete_zeros_performed=False,full_qualification=False,production_gates_opened=False,
  formal_Goodput=False,performance_measured=False,adopted=False)
