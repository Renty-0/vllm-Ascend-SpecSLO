"""One new target mapping; actual current-engine qualification owns permission."""
from contextlib import contextmanager
from base173 import BASE,functions,inherited_module,require
DOMAIN=(16,80,'causal')
KEY_TAIL='|steps:1|fia:exact-length-task-update|buffers:(((16, 32), (16, 32), (2048, 2048)),)'
GREEDY=('target-greedy:151936'+KEY_TAIL,(80,))
HIDDEN=('target-hidden:0'+KEY_TAIL,(80,))
_ISSUER=object()

def current(engine):
 proof=getattr(engine,'_v173_causal80_permission',None)
 require(proof is not None and proof['issuer'] is _ISSUER and proof['engine'] is engine
  and proof['evidence'] is getattr(engine,'_v173_actual_causal80',None)
  and proof['evidence']['audit']['math_independent_qualified'] is True,'own actual causal80 issuer')
 if not engine.is_draft:
  for key in (GREEDY,HIDDEN):
   entry=engine.graph_runner.target_entries.get(key)
   require(entry is not None and entry.runtime_validated is True and key not in engine.graph_runner.disabled_entry_keys,
    'actual original naturally closed causal80 greedy and hidden entries')
 return proof

def issue(engine,evidence):
 require(evidence is getattr(engine,'_v173_actual_causal80',None)
  and evidence['audit']['math_independent_qualified'] is True
  and engine.cache_allocation is None and engine.cache_block_tables is None
  and not engine.graph_runner.graph_cache_sealed,'released own actual component before causal80 activation')
 require(not hasattr(engine,'_v173_causal80_permission'),'current causal80 issuer once')
 engine._v173_causal80_permission=dict(issuer=_ISSUER,engine=engine,evidence=evidence)
 current(engine)

def expected_geometry(n,q,f):
 if (n,q,f)==DOMAIN:return 80,16,[]
 from candidate_controls_v132 import MAPPER_PARTITIONS
 from complete_search import envelope as complete
 from target_layout import legacy_envelope,STATIC_NATIVE_KEYS,_lengths
 if (n,q,f) in MAPPER_PARTITIONS:return complete(n,q,f)
 original=legacy_envelope(n,q)
 if (*original[:2],f) in STATIC_NATIVE_KEYS:return original
 for m,s in (((96,32),) if n<=5 else ((96,38),(128,38),(165,38))):
  pad=_lengths(n,q,m,s)
  if pad is not None:return m,s,pad
 raise RuntimeError('No unchanged original mapper extent')

@contextmanager
def qualification_geometry(engine):
 require(getattr(engine,'_v173_qualified_component',None) is not None
  and engine.cache_allocation is not None and not engine.graph_runner.graph_cache_sealed
  and not hasattr(engine,'_v173_qualification_geometry'),'bounded current own-B4 qualification lane only')
 engine._v173_qualification_geometry=_ISSUER
 try:yield
 finally:
  require(engine._v173_qualification_geometry is _ISSUER,'qualification lane identity')
  del engine._v173_qualification_geometry

def dispatch(control,n,q,f):
 if (n,q,f)==DOMAIN and control.engine is not None and getattr(control.engine,'_v173_qualification_geometry',None) is _ISSUER:
  control.engine._v173_qualification_mapper_calls=getattr(control.engine,'_v173_qualification_mapper_calls',0)+1
  return 80,16,[]
 if (n,q,f)==DOMAIN and control.engine is not None and hasattr(control.engine,'_v173_causal80_permission'):
  current(control.engine)
  control.engine._v173_causal80_mapper_calls=getattr(control.engine,'_v173_causal80_mapper_calls',0)+1
  return 80,16,[]
 from candidate_controls_v132 import MAPPER_PARTITIONS
 from complete_search import envelope as complete
 return complete(n,q,f) if (n,q,f) in MAPPER_PARTITIONS else control.original_envelope(n,q,f)

def controls_class():
 original=inherited_module('candidate_controls_v132');ns=dict(vars(original),dispatch=dispatch)
 functions(BASE/'candidate_controls_v132.py',ns,changes=((
  'def mapper(n,q,f):return complete(n,q,f) if (n,q,f) in MAPPER_PARTITIONS else self.original_envelope(n,q,f)',
  'def mapper(n,q,f):return dispatch(self,n,q,f)',1),))
 return ns['CandidateControls']
