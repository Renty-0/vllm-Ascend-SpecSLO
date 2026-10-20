"""Original static TP3 route contract extended only to existing row profiles."""
import hashlib,json
from pathlib import Path
from scope151 import MC2_ROWS,require
PROFILE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/mc2-profile.json')
RAW_SHA='ad9e0648810ed697679ee84b8bbc2939ea14bfb687d0215d27f403ad5526db1c'
OPERATOR='tp3_native_matmul+allreduce_add_rmsnorm'
def check(e):
 from vllm_ascend.spec_decode.pearl.mc2 import _mc2_profile_sha256
 p=e.model.config.pearl_mc2_profile;m=getattr(e,'mc2_static_route_manifest',None) or getattr(e.model,'_mc2_static_route_manifest',None)
 require(hashlib.sha256(PROFILE.read_bytes()).hexdigest()==RAW_SHA and p.metadata.get('operator')==OPERATOR
  and p.metadata.get('standalone_chained_flush') is True and len(e.model.layers)==64 and m is not None
  and len(m.routes)==128 and m.profile_sha256==_mc2_profile_sha256(p) and bool(getattr(e.model,'_mc2_static_routes_frozen',False))
  and set(MC2_ROWS)<=set(e.model._mc2_chained_row_counts),'original raw profile/static128 route identity')
 require(all(r.enabled and r.operator_kind==OPERATOR and all(n in r.decisions and r.decisions[n].qualified for n in MC2_ROWS) for r in m.routes),
  'all actual original row profiles qualified across128 TP3 projections')
 return dict(passed=True,raw_profile_sha256=RAW_SHA,canonical_profile_sha256=_mc2_profile_sha256(p),routes=128,rows=list(MC2_ROWS),new_profiles_created=False)
