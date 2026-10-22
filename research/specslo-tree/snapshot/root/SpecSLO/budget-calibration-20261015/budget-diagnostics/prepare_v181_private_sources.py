"""HOST source preparation only; never imports a backend or edits a parent."""
import ast, hashlib, json, re
from pathlib import Path

B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-parent3-repair-frontier-union3-active32-full-v181'
OLD=B/'tree-high-normal-cap3-active32-full-v176'
U=B/'tree-live-shared4-union-full-v168'
L=B/'tree-live-shared4-union-v159'
V151=B/'tree-normal4-parent3-shared4-eager-v151'
V152=B/'tree-union-shared4-protocol-local-v152'
V154=B/'tree-natural-positive-union-discovery-local-v154'
V161=B/'tree-union-own-epoch-metadata-fix-local-v161'
V179=B/'tree-linear-parent4-child3-bank-binding-full-v179'
records={}
pending={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(name,text):
 ast.parse(text)
 path=P/name
 if path.exists():raise RuntimeError('preserve existing candidate source: '+str(path))
 pending[name]=text
 return path
def project(name,parent,changes=()):
 parent=Path(parent);original=parent.read_text();text=original;rows=[]
 for old,new,count in changes:
  actual=text.count(old)
  if actual!=count:raise RuntimeError(f'{name}: anchor count {actual}!={count}: {old!r}')
  text=text.replace(old,new);rows.append(dict(old=old,new=new,count=count))
 emit(name,text);records[name]=dict(parent=str(parent),parent_sha256=sha(parent),changes=rows,sha256=hashlib.sha256(text.encode()).hexdigest())
 return text
def rename(parent,old,new):
 s=Path(parent).read_text();return (old,new,s.count(old))

def prepare():
 # The production runner and policy are exactly P176 with private epoch names.
 for stem in ('base','source','progress','runtime','trace','scope','audit','component','controls','shape','observe'):
  parent=OLD/f'{stem}176.py';changes=[rename(parent,'176','181')]
  if stem=='base':
   changes += [("ACTION='high-normal-cap3-active32-full463'","ACTION='parent3-repair-frontier-union3-active32-full463'",1),
    ("for root in (BASE,ACTIVE):","PROTO=B/'tree-union-shared4-protocol-local-v152'\nV151=B/'tree-normal4-parent3-shared4-eager-v151'\nV154=B/'tree-natural-positive-union-discovery-local-v154'\nV161=B/'tree-union-own-epoch-metadata-fix-local-v161'\nV159=B/'tree-live-shared4-union-v159'\nV149=B/'tree-repair-frontier-fusion-local-v149'\nfor root in (BASE,ACTIVE,V151,PROTO,V154,V161,V159,V149):",1)]
  if stem=='component':
   changes += [("'future_full_requires_21scope26phase_and_two65zero=True',","'future_full_requires_21scope26phase_and_two65zero=True',",1),
    ("'future_full_requires_standalone_scope_and_two65zero=True',1),))","'future_full_requires_standalone_scope_and_two65zero=True',1),(\n  'from import_origins_v132 import require_loaded_origins as assert_origins',\n  'from extension181 import loaded_origins as assert_origins',1)))",1),
    ("engine._v181_qualified_component=proof\n return proof","from component_union181 import run as run_union_component\n union=run_union_component(engine,native,stage,policy,output)\n proof=dict(proof,union_component=union)\n engine._v181_qualified_component=proof\n return proof",1)]
  if stem=='audit':
   changes += [("require(component['component_numerical_qualified'] is True,'own retained original B4 component')","require(component['component_numerical_qualified'] is True,'own retained original B4 component')\n from component_union181 import audit_saved as audit_union\n union=audit_union(directory/'new-parent3-union3-own-full527')",1),
    ("and r['component']['audit']==component,'actual201/44/114 plus current B4 proof')","and r['component']['audit']==component and r['component']['union_component']['audit']==union,'actual201/44/114 plus current B4 and new union3 proof')",1),
    ("original_B4_component=component,old_21scope26phase_executed=False,","original_B4_component=component,new_union3_component=union,old_21scope26phase_executed=False,",1)]
  project(f'{stem}181.py',parent,changes)
 # Exact full527 bank includes the already reviewed missing-hashlib repair;
 # no parent4 algorithms or capability are imported from P179.
 parent=V179/'bank179.py'
 project('bank181.py',parent,[rename(parent,'179','181')])
 # Diagnostic observer retains the tested actual interior-padding inspection.
 parent=U/'observer168.py';project('observer181.py',parent,[rename(parent,'168','181')])
 # New numerical representatives have true three-step draft output and ledger.
 parent=U/'pure168.py'
 project('pure181.py',parent,[rename(parent,'168','181'),
  ("Case('N32-E32-R32',32,32)","Case('N16-E16-R16',16,16)",1),
  ('NORMAL_B=(4,)*19+(3,)+(2,)*5+(1,)*7','NORMAL_B=(3,)*16',1),
  ('normal=list(NORMAL_B) if c.normal==32 else [4]*c.normal','normal=list(NORMAL_B) if c.normal==16 else [3]*c.normal',1),
  ('range(128,133 if i<c.normal else 136)','range(128,132 if i<c.normal else 135)',1),
  ('len(seen[\'draft\'][0][\'hidden\'])==4','len(seen[\'draft\'][0][\'hidden\'])==3',1),
  ("(M,4,2)","(M,3,2)",1),
  ('prefixes,roots,4,M)','prefixes,roots,3,M)',1),
  ("(encoded,c.count,4)","(encoded,c.count,3)",1),
  ('/8 for s','/6 for s',1),
  ('[root,*row[:3]]','[root,*row[:2]]',1),
  ('fourstep','threestep',parent.read_text().count('fourstep')),
  ('four-step','three-step',parent.read_text().count('four-step'))])
 # Natural protocol source uses 16 normal +16 eager owners, original scratch.
 parent=V151/'source151.py'
 project('epoch181.py',parent,[
  ('from scope151 import require,digest','from math181 import require,digest',1),
  ('list(range(64))+list(range(64,102))','list(range(32))+list(range(64,102))',1),
  ('e._cache_storage_sequence_capacity==128 and len(tables)==128','e._cache_storage_sequence_capacity==527 and len(tables)==527',1),
  ('len(pages)==len(set(pages))==166','len(pages)==len(set(pages))==2*32+38',1),
  ('actual128 storage/166 independent real+scratch pages/544 pool','actual527 storage/derived102 independent real+scratch pages/544 pool',1),
  ('storage=128,private_scratch','storage=527,private_scratch',1),
  ('real_owners=list(range(64))','real_owners=list(range(32))',1),
  ('len(contexts)==len(repairs)==64','len(contexts)==len(repairs)==32',1),
  ('e._ensure_cache_capacity(list(range(64))','e._ensure_cache_capacity(list(range(32))',1)])
 parent=V154/'provider154.py'
 project('provider181.py',parent,[
  ('from scope154 import P,require,digest','from math181 import require,digest\nfrom base181 import V154 as P',1),
  ('len(original)==64','len(original)==32',1),
  ('range(32,48)','range(16,24)',2),
  ('(owner-32)%8','(owner-16)%8',1),
  ('range(48,64)','range(24,32)',1)])
 # Existing P161 own-allocation metadata gate, with the exact new owner domain.
 parent=V161/'metadata161.py'
 project('metadata181.py',parent,[
  ('case.normal==case.eager==32','case.normal==case.eager==16',1),
  ('case.union_M==36','case.union_M==20',1),
  ('range(32,64)','range(16,32)',2),
  ('[case.prefix+3]*32','[case.prefix+3]*16',1)])
 # The own same-U protocol oracle follows its own history. Only the mixed
 # shared chain changes4->3; parent3 and next-normal3 keep their original ABI.
 parent=V152/'draft152.py';text=parent.read_text()
 constant_nodes=ast.parse((V161/'draft161.py').read_text()).body;constants={}
 for node in constant_nodes:
  if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
   try:constants[node.targets[0].id]=ast.literal_eval(node.value)
   except (ValueError,TypeError):pass
 changes=[('from scope152 import require,digest,bucket','from math181 import require,digest,bucket',1),
  ('from reuse152 import source_contexts','from provider181 import source_contexts\nfrom metadata181 import capture_actual,compare_fresh',1),
  ('import source152 as epoch','import epoch181 as epoch',1)]
 for k in ('ACTUAL','COMPARE','SIGNATURE','META'):changes.append((constants[k+'_OLD'],constants[k+'_NEW'],1))
 changes += [('range(32,64)','range(16,32)',1),('range(32)','range(16)',3),
  ('[p]*32','[p]*16',1),('[p]*64','[p]*32',2), # Specific parent occurrence is repaired below if ambiguous.
 ]
 # Remaining exact mixed/parent occurrence transforms are populated after
 # the initial materialization inspection, rather than broad numeric rewrite.
 project('draft181.py',parent,changes)
 # Reuse original target verifier/compaction algorithms at the smaller cohort.
 parent=V151/'target151.py'
 project('target181.py',parent,[
  ('from scope151 import require,target_shape','from math181 import require,target_shape',1),
  ('import source151 as epoch','import epoch181 as epoch',1),
  ('[0]*64','[0]*32',1),('range(64)','range(32)',1)])
 parent=V151/'budget151.py'
 project('budget181.py',parent,[
  ('from scope151 import require','from math181 import require',1),
  ('range(32,64)','range(16,32)',1),('range(64)','range(32)',2),
  ('[False]*32+[True]*32','[False]*16+[True]*16',1),
  ('==result.future_eager_candidates==96','==result.future_eager_candidates==48',1),
  ('normal_candidates=96,future_eager_candidates=96,selected_total=192,actual_work_rows=64','normal_candidates=48,future_eager_candidates=48,selected_total=96,actual_work_rows=32',1)])
 parent=V151/'protocol151.py'
 project('protocol181.py',parent,[
  ('from scope151 import require','from math181 import require',1),
  ('import source151 as epoch','import epoch181 as epoch',1),('import target151 as target','import target181 as target',1),
  ('from budget151 import work_plan,refine','from budget181 import work_plan,refine',1),
  ('import next151','import next181',1),('next151.proof','next181.proof',1),
  ('range(32,64)','range(16,32)',5),('range(32)','range(16)',2),('range(64)','range(32)',2),
  ('i<32','i<16',2),('i>=32','i>=16',1),('[0]*64','[0]*32',1),
  ("outputs,accepted,_=stage('original-verdict-broadcast',lambda:e._broadcast_spec_rhythm_tree_verdict(observed,plans,0.))", "outputs,accepted,_=stage('original-verdict-broadcast',lambda:e._broadcast_spec_rhythm_tree_verdict(observed,plans,0.))\n if source.get('_v154_before_parent_commit') is not None:\n  source['_v154_before_parent_commit'](outputs,accepted)",1)])
 parent=V151/'next151.py'
 project('next181.py',parent,[
  ('from scope151 import require,bucket','from math181 import require,bucket',1),
  ('import draft151 as draft','import draft181 as draft',1),('import source151 as epoch','import epoch181 as epoch',1),
  ('range(64)','range(32)',1),('(0,32)','(0,16)',1),('ids[start:start+32]','ids[start:start+16]',1),
  ('[case.repair]*32+[0]*32','list(case.repair_vector)',2),('[128]*64','[128]*32',5),
  ('selected_candidates=96,actual_shared_steps=3,real_raw_nodes=192,returned_ABI_slots=256','selected_candidates=48,actual_shared_steps=3,real_raw_nodes=96,returned_ABI_slots=128',1)])
 # Concrete original natural-positive discovery/real commit/next controller.
 parent=V154/'actor154.py'
 project('natural181.py',parent,[
  ('from scope154 import require,case,contract,NATIVE_COUNTERS,FAILURES,digest','from math181 import require,case,contract,NATIVE_COUNTERS,FAILURES,digest',1),
  ('import draft154 as draft','import draft181 as draft',1),('import source154 as epoch','import epoch181 as epoch',1),
  ('import protocol154 as protocol','import protocol181 as protocol',1),('import target154 as target','import target181 as target',1),
  ('from provider154 import evidence','from provider181 import evidence',1),
  ('import next154','import next181',1),('next154.proof','next181.proof',1),
  ('range(32,64)','range(16,32)',3),('range(32)','range(16)',1),('range(64)','range(32)',1),
  ('[0]*64','[0]*32',1),
  ('staged_dependency=list(dependency),staged_frontier=eager[\'eager_frontier_token\'],','staged_dependency=list(dependency),staged_frontier=eager[\'eager_frontier_token\'],staged_parent_proposal_id=eager[\'eager_parent_proposal_id\'],staged_dependency_length=length,',1)])
 for name,text in pending.items():(P/name).write_text(text)
 (P/'DERIVATIONS181.json').write_text(json.dumps(records,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':prepare()
