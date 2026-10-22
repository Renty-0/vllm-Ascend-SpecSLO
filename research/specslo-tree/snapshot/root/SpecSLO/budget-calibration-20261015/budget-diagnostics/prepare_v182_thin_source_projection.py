"""HOST source preparation only. Do not execute during ROOT quiet interval."""
from pathlib import Path
import ast,json,hashlib
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-compact-b3-target-union3-active32-full-v182';U=B/'tree-parent3-repair-frontier-union3-active32-full-v181';A=B/'tree-uniform-b3-three-bucket-full-v180';G=D/'v182-root-execution'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows={}
def project(name,parent,changes):
 text=parent.read_text()
 for old,new,count in changes:
  assert text.count(old)==count,(name,old[:80],text.count(old));text=text.replace(old,new)
 target=P/name;assert not target.exists(),target;target.write_text(text)
 rows[name]=dict(parent=str(parent),parent_sha256=sha(parent),changes=[dict(old=x,new=y,count=n)for x,y,n in changes],sha256=sha(target))
def namespace(name):
 parent=U/(name+'181.py');project(name+'182.py',parent,[('181','182',parent.read_text().count('181'))])
for name in ('base','source','progress','runtime','trace','observe'):namespace(name)
base=P/'base182.py';text=base.read_text();anchor="for root in (BASE,ACTIVE,V151,PROTO,V154,V161,V159,V149):";new="P180=B/'tree-uniform-b3-three-bucket-full-v180'\nP181=B/'tree-parent3-repair-frontier-union3-active32-full-v181'\nfor root in (BASE,ACTIVE,V151,PROTO,V154,V161,V159,V149,P180,P181):"
assert text.count(anchor)==1;text=text.replace(anchor,new)
extra='''
def parent_module(name,root):
 import importlib
 module=importlib.import_module(name);path=Path(root)/(name+'.py')
 require(Path(vars(module)['__file__']).resolve()==path,'exact immutable helper source '+name)
 checked(path);return module
'''
text+=extra;base.write_text(text);rows['base182.py']['changes'].append(dict(old=anchor,new=new,count=1));rows['base182.py']['append_text']=extra;rows['base182.py']['sha256']=sha(base)
run=P/'runtime182.py';text=run.read_text();old="partial_union=getattr(engine,'_v182_partial',{}),partial_natural=getattr(engine,'_v151_partial',{})";new="partial_union=getattr(engine,'_v181_partial',{}),partial_geometry=getattr(engine,'_v180_geometry_partial',{}),partial_natural=getattr(engine,'_v151_partial',{})";assert text.count(old)==1;run.write_text(text.replace(old,new));rows['runtime182.py']['changes'].append(dict(old=old,new=new,count=1));rows['runtime182.py']['sha256']=sha(run)
# Baseline: exact original component prefix, one origin callback import change.
s=(U/'component181.py').read_text();cut=" from component_union181 import run as run_union_component\n union=run_union_component(engine,native,stage,policy,output)\n proof=dict(proof,union_component=union)\n engine._v181_qualified_component=proof\n return proof\n"
assert s.count(cut)==1
project('baseline182.py',U/'component181.py',[('from base181 import','from base182 import',1),('from extension181 import loaded_origins as assert_origins','from extension182 import loaded_origins as assert_origins',1),(cut,' return proof\n',1)])
project('component_union182.py',U/'component_union181.py',[('from base181 import require','from base182 import require',1),(' import natural181',' import natural_joint182 as natural181',1),("  diagnostic_inventory=[r['inventory'] for r in reports],representatives=expected,","  joint_target=__import__('audit_joint182').reports(reports),\n  diagnostic_inventory=[r['inventory'] for r in reports],representatives=expected,",1)])
project('audit182.py',U/'audit181.py',[('181','182', (U/'audit181.py').read_text().count('181'))])
audit=P/'audit182.py';text=audit.read_text();changes=[(" union=audit_union(directory/'new-parent3-union3-own-full527')"," union=audit_union(directory/'new-parent3-union3-own-full527')\n from audit_component180 import audit as audit_geometry\n geometry=audit_geometry(directory/'uniformB3-three-bucket-own-current-process')",1),
 ("r['component']['union_component']['audit']==union","r['component']['union_component']['audit']==union and r['component']['uniformB3']['audit']==geometry\n   and r['component']['shared_current_base']['same_current_base_object'] is True\n   and r['component']['shared_current_base']['original_B4_invocations']==1",1),
 ("n,q,f=c['N'],c['real_query_count'],c['family'];m,s,padding=(mapper if (n,q,f) in MAPPER_PARTITIONS else old_mapper)(n,q,f)","n,q,f=c['N'],c['real_query_count'],c['family']\n   from geometry180 import extent\n   compact=extent(n) if c['budgets']==[3]*n and f=='causal' else None\n   if compact is not None:m,s=compact;padding=[4]*(s-n)\n   else:m,s,padding=(mapper if (n,q,f) in MAPPER_PARTITIONS else old_mapper)(n,q,f)",1),
 ("original_B4_component=component,new_union3_component=union,old_21scope26phase_executed=False,","original_B4_component=component,new_union3_component=union,new_compact_target_component=geometry,old_21scope26phase_executed=False,",1)]
for old,new,count in changes:
 assert text.count(old)==count,(old,text.count(old));text=text.replace(old,new)
audit.write_text(text);rows['audit182.py']['changes']+=[dict(old=o,new=n,count=c)for o,n,c in changes];rows['audit182.py']['sha256']=sha(audit)
(P/'DERIVATIONS182.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
# No parent archive copied, no model, test, installation or NPU call here.
print(json.dumps(dict(candidate=str(P),projected_files=len(rows),NPU_used=False,freeze_created=False)))
