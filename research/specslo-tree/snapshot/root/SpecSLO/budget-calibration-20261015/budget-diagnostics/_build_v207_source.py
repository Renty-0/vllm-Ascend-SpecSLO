from pathlib import Path
import ast,copy,difflib,hashlib,json,re
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-normal-cap3-original-geometries-full-v207';G=D/'v207-root-execution'
OLD=B/'tree-empty-active-cohort-causal80-active32-full-v197'
P180=B/'tree-uniform-b3-three-bucket-full-v180';P176=B/'tree-high-normal-cap3-active32-full-v176'
assert not P.exists() and not G.exists();P.mkdir();G.mkdir()
ACTION='normal-cap3-original-geometries-full463'
RECORD={}
def ns(s,old,new):
 keys={}
 def k(m):
  x='__HASH_'+str(len(keys))+'__';keys[x]=m[0];return x
 s=re.sub(r'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])',k,s).replace(str(old),str(new))
 for a,b in keys.items():s=s.replace(a,b)
 return s

def derive(src,dst,old):
 original=src.read_text();text=ns(original,old,207)
 if old==197:
  text=text.replace('tree-empty-active-cohort-causal80-active32-full-v207',P.name).replace('empty-active-cohort-causal80-active32-full463',ACTION)
 (dst).write_text(text);RECORD[str(dst)]=(src,old);return text

def put(name,s): (P/name).write_text(s)
def change(name,old,new,count=1):
 p=P/name;s=p.read_text();assert s.count(old)==count,(name,old,s.count(old));p.write_text(s.replace(old,new))
for src in sorted(OLD.glob('*.py')):
 if src.name in ('test_full197.py','host_full197.py','check197.py'):continue
 derive(src,P/src.name.replace('197','207'),197)
for src in sorted((D/'v197-root-execution').iterdir()):
 if src.name in ('guard197.py','launch197.sh','startup197.sh'):derive(src,G/src.name.replace('197','207'),197)
# Sole production/source parent197, original P148 permit remains unchanged.
p=P/'base207.py';p.write_text(p.read_text()+'''\nPARENT=B/'tree-empty-active-cohort-causal80-active32-full-v197'
if str(PARENT) not in sys.path:sys.path.append(str(PARENT))
def production_parent(name):
 import importlib
 module=importlib.import_module(name)
 require(Path(vars(module)['__file__']).resolve()==PARENT/(name+'.py'),'sole frozen197 production source')
 checked(PARENT/(name+'.py'));return module
''')
# Exact original shape algorithm, only high-N individual ceiling4->3 and names.
derive(P176/'shape176.py',P/'shape207.py',176)
derive(P176/'observe176.py',P/'observe207.py',176)
put('controls207.py','''"""Original197 mapper and three config deltas; cap3 policy needs this engine proof."""
from base207 import P,BASE,functions,inherited_module,require
_CLASS=None
def controls_class():
 global _CLASS
 if _CLASS is None:
  import shape207
  from mapper173 import dispatch
  original=inherited_module('candidate_controls_v132');ns=dict(vars(original),shape=shape207.shape,dispatch=dispatch)
  functions(BASE/'candidate_controls_v132.py',ns,changes=((
   'normal_N_gt8_cap=4','normal_N_gt8_cap=3',1),(
   'def mapper(n,q,f):return complete(n,q,f) if (n,q,f) in MAPPER_PARTITIONS else self.original_envelope(n,q,f)',
   'def mapper(n,q,f):return dispatch(self,n,q,f)',1),(
   'def candidate_shape(instance,**kwargs):return shape(policy,instance,**kwargs)',
   "def candidate_shape(instance,**kwargs):\\n            from capability207 import current\\n            current(self.engine)\\n            return shape(policy,instance,**kwargs)",1)),filename=str(P/'controls207.py')+'::same197-mapper-cap3-class')
  require(ns['CandidateControls'].__init__.__globals__['shape'] is shape207.shape,'actual new shape binding')
  require(ns['filter_eager_parents'] is original.filter_eager_parents,'original parent3 gate unchanged')
  _CLASS=ns['CandidateControls']
 return _CLASS
''')
# Variable-N genuine B3 math helpers, narrowed to one actual old mapper case.
derive(P180/'bank180.py',P/'bank207.py',180)
s=(P/'bank207.py').read_text();start=s.index('def cases():');end=s.index('\nclass DualBank:',start)
s=s[:start]+'''def cases():
 from scope_v146 import cases as original
 from mapper173 import expected_geometry
 source=copy.deepcopy(original()[0]);owners=copy.deepcopy(source['owners'][:16])
 for i,row in enumerate(owners):row['owner']=i;row['reference_owner']=16+i
 m,s,pad=expected_geometry(16,64,'causal')
 require((m,s,pad)==(96,32,[2]*16),'original197 N16 B3 is96/S32, never128')
 return [dict(index=0,N=16,C=48,Q=64,budgets=[3]*16,owners=owners,route='old',name='original96-B3',M=m,S=s,dummy_lengths=pad,
  source_path=source['source_path'],source_sha256=source['source_sha256'],scratch_per_bank=16,
  initial_root_teacher_forced_not_own_AR=True,subsequent_draft_and_verdict_actual=True)]
'''+s[end:]
s=s.replace('from geometry207 import extent','').replace('9<=n<=24','n==16')
put('bank207.py',s)
derive(P180/'draft180.py',P/'draft207.py',180)
derive(P180/'target180.py',P/'target207.py',180)
s=(P/'target207.py').read_text().replace('import geometry207','')
start=s.index(' if not reference:');end=s.index(' else:\n  # Reference',start)
s=s[:start]+''' if not reference:
  pp,rr,dd,ids,real=engine._pad_spec_rhythm_target_tree_graph(plans,roots,rows,list(range(n)),list(range(n)),work['logical'])
'''+s[end:]
# No new geometry helper/mapper; actual parent production pad above supplies dummy.
put('target207.py',s)
# One current197 component, plus sameM B3 current proof; math not inherited.
change('component207.py','def run(engine,native,stage,policy,output):','def original197_run(engine,native,stage,policy,output):')
p=P/'component207.py';p.write_text(p.read_text()+'''\ndef run(engine,native,stage,policy,output):
 parent=original197_run(engine,native,stage,policy,output)
 from actor207 import run as run_B3
 extra=run_B3(engine,native,stage,policy,output)
 result=dict(parent,current_B3=extra)
 engine._v207_qualified_component=result
 from capability207 import issue
 stage('issue-current-cap3-original-geometries',lambda:issue(engine,result,extra))
 return result
''')
change('extension207.py',"return entry_namespace(output_directory),parent_module('mapper173').controls_class()","from controls207 import controls_class\n return entry_namespace(output_directory),controls_class()")
change('extension207.py',"new_hot_collectives=0)","new_hot_collectives=0,normal_N_gt8_cap=3,normal_N_le8_cap=5,eager_shared_steps_max=3,production_new_keys=0,diagnostic_new_hidden_keys=1)")
change('trace207.py',"from candidate_phases_v132 import split_snapshot","from observe207 import split_snapshot")
change('trace207.py',"functions(ACTIVE/'pair160.py',ns,names=('run_pair',))","functions(ACTIVE/'pair160.py',ns,names=('run_pair',),changes=((\n  'from candidate_phases_v132 import split_snapshot, split_delta, update_snapshot',\n  'from observe207 import split_snapshot, split_delta, update_snapshot',1),))")
# Scope checks all current new evidence; original197 math/prefill remains intact.
change('scope207.py',"'same current independent multiowner+next4 proof')","'same current independent multiowner+next4 proof')\n  from capability207 import current\n  current(engine)")
change('scope207.py',"constructor_coalesce_unchanged=True)","constructor_coalesce_unchanged=True,normal_N_gt8_cap=3,normal_N_le8_cap=5,original_mapper_unchanged=True,production_new_keys=0,diagnostic_hidden_keys=1,current_B3_math=True)")
# Parent baseline startup marker remains actual173; local marker207 only report.
# Runtime untouched algorithm/clock and strict single constructor.
# Source maps and whole text projections written after remaining new sources.
(P/'_BUILD_DONORS.json').write_text(json.dumps({k:[str(a),b]for k,(a,b)in RECORD.items()},indent=2))
