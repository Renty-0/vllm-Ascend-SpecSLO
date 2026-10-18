"""Seven new real-N usages of existing80; original197 dispatch outside them."""
from contextlib import contextmanager
from base209 import require,parent_module
from mapper173 import GREEDY,HIDDEN
_ISSUER=object()
DOMAIN=frozenset((n,5*n,'causal')for n in range(9,16))

def current(e):
 p=getattr(e,'_v209_usage_permission',None)
 require(p is not None and p['issuer']is _ISSUER and p['engine']is e
  and p['evidence']is getattr(e,'_v209_actual_usage',None)
  and p['evidence']['audit']['math_independent_qualified']is True,'current same-engine seven-use math/protocol proof')
 require(getattr(e,'_v209_parent_component',None)is p['parent'],'same current full197 component')
 if not e.is_draft:
  for k in(GREEDY,HIDDEN):
   a=e.graph_runner.target_entries.get(k)
   require(a is p['entries'][k]and a.runtime_validated is True and k not in e.graph_runner.disabled_entry_keys,'exact retained80 entry objects and natural flags')
 return p

def active(e):
 if getattr(e,'_v209_usage_geometry',None)is _ISSUER:return True
 if hasattr(e,'_v209_usage_permission'):current(e);return True
 return False

def dispatch(control,n,q,f):
 e=control.engine
 if (n,q,f)in DOMAIN and e is not None and active(e):
  from target_layout import _lengths
  pad=_lengths(n,q,80,16);require(pad==[5]*(16-n),'original legal dummy partition')
  return 80,16,pad
 return parent_module('mapper173').dispatch(control,n,q,f)

def controls_class():
 Parent=parent_module('mapper173').controls_class()
 class Controls(Parent):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs)
   self.mapper=lambda n,q,f:dispatch(self,n,q,f)
   self.mapper._v209_control=self
 return Controls

@contextmanager
def qualification_geometry(e):
 require(getattr(e,'_v209_parent_component',None)is not None and e.cache_allocation is not None
  and not e.graph_runner.graph_cache_sealed and not hasattr(e,'_v209_usage_geometry'),'bounded current197-only qualification context')
 e._v209_usage_geometry=_ISSUER
 try:yield
 finally:
  require(e._v209_usage_geometry is _ISSUER,'same bounded context')
  del e._v209_usage_geometry

def check_frame(e,plans,roots,rows,ids,frame):
 from target_layout import plan_family
 n=len(plans)
 require((n,sum(p.candidate_budget+1 for p in plans),plan_family(plans))in DOMAIN
  and all(p.candidate_budget==4 and len(r)==4 for p,r in zip(plans,rows,strict=True)),'real current allB4 plans/rows, not fake16')
 pp,rr,dd,ss,real=frame;pad=list(e._tree_graph_scratch_sequence_ids[:16-n]);tables=e.cache_allocation.block_tables
 require(e.cache_allocation is not None and e.cache_block_tables is not None and len(tables)==527
  and len(ids)==len(set(ids))==n and set(ids).isdisjoint(pad)
  and ss==list(ids)+pad and real==n and len(pp)==len(rr)==len(dd)==len(ss)==16
  and all(a is b for a,b in zip(pp[:n],plans,strict=True))and rr[:n]==list(roots)and dd[:n]==list(rows),'actual lease and real prefix/owner/object identities')
 for seq,p,r,d in zip(ss[n:],pp[n:],rr[n:],dd[n:],strict=True):
  require(p.width==1 and p.depth==4 and p.prefix_len==0 and p.candidate_budget==4
   and p.cache_positions.tolist()==list(range(5))and r==0 and d==[0]*4
   and tables[seq][0]>=0,'original independently allocated scratch q5 dummy')
 require(all(0<=i<len(tables)and all(type(v)is int and -1<=v<544 for v in tables[i])for i in ss),'actual selected allocation page bounds')
 return frame

def install_pad(native):
 old=native.NativePearlEngine._pad_spec_rhythm_target_tree_graph
 def padded(e,plans,roots,rows,ids,active_ids,states):
  from target_layout import plan_family,envelope
  control=getattr(envelope,'_v209_control',None)
  selected=(control is not None and envelope is control.mapper
   and control.engine is e and control.mode=='candidate')
  eligible=selected and 9<=len(plans)<=15 and all(p.candidate_budget==4 for p in plans) and plan_family(plans)=='causal'
  enabled=eligible and active(e)
  value=old(e,plans,roots,rows,ids,active_ids,states)
  return check_frame(e,plans,roots,rows,ids,value)if enabled else value
 padded._v209_original=old
 native.NativePearlEngine._pad_spec_rhythm_target_tree_graph=padded
 return old

def issue(e,evidence):
 require(evidence is getattr(e,'_v209_actual_usage',None)and evidence['audit']['math_independent_qualified']is True
  and e.cache_allocation is None and e.cache_block_tables is None and not e.graph_runner.graph_cache_sealed
  and not hasattr(e,'_v209_usage_geometry')and not hasattr(e,'_v209_usage_permission'),'released current usage evidence before production issuer')
 entries={}if e.is_draft else{k:e.graph_runner.target_entries[k]for k in(GREEDY,HIDDEN)}
 e._v209_usage_permission=dict(issuer=_ISSUER,engine=e,evidence=evidence,parent=e._v209_parent_component,entries=entries)
 current(e)

def expected_geometry(n,q,f):
 if(n,q,f)in DOMAIN:return 80,16,[5]*(16-n)
 return parent_module('mapper173').expected_geometry(n,q,f)
