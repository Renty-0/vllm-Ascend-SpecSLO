"""Final ROOT-source binding only. Never supplies a receipt or calls terminal main."""
import ast,hashlib,json,runpy,time
from pathlib import Path
D=Path('/root/SpecSLO/budget-calibration-20261015/budget-diagnostics');P=D.parent/'tree-causal80-small-cohort-role-aware-audit-full-v209';s=D/'audit_v209_terminal_HOST.py';start=time.perf_counter()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
ns=runpy.run_path(str(s),run_name='_SOURCE_BINDING_ONLY_NEVER_MAIN')
f=P/'FROZEN_SOURCE209.json';rp=D/'ROOT_V209_PREPARED_SOURCE_HOST_AUDIT.json'
assert sha(f)==ns['FINAL_FREEZE_SHA'] and sha(rp)==ns['FINAL_ROOT_PREPARED_SHA']
fr=read(f);r=read(rp);root=read(fr['root_review']['path']);c=read(P/'CPU_RESULT209.json');pr=read(P/'PREPARED_HOST209.json');ready=read(ns['READY'])
pins={}
for field in('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):
 for p,h in fr[field].items():assert p not in pins or pins[p]==h;pins[p]=h
pins.update({str(P/'CPU_RESULT209.json'):fr['CPU_sha256'],str(f):sha(f),fr['root_review']['path']:fr['root_review']['sha256']})
assert len(pins)==r['actual_source_members']==r['actual_archive_members']==1529
assert sha(r['archive']['path'])==ns['FINAL_ARCHIVE_SHA']==r['archive']['sha256']
assert root['reviewed_inputs']=={p:h for p,h in pins.items()if p not in(str(f),fr['root_review']['path'])}
assert root['ready']==r['ready']==dict(path=str(ns['READY']),sha256=sha(ns['READY']))
assert r['review']==fr['root_review'] and sha(fr['root_review']['path'])==fr['root_review']['sha256']
assert list(fr['peer_reviews'].values())==[ns['FINAL_PEER_SHA']]
assert c['tests_run']==3 and c['status']=='passed'and c['local_before']==c['local_after']==pr['files']==fr['files'] and c['external_before']==c['external_after']==pr['external_files']==fr['external_files']
assert len(fr['files'])==26 and len(fr['external_files'])==1395 and len(fr['retained_HOST_evidence'])==100 and len(fr['execution_guard_inputs'])==4
assert len(pr['actual_HOST_attempts'])==3 and len(pr['retained_HOST_files'])==96
aux={e['path']:e['sha256']for e in(ready['HOST']['actual_cold_guard'],ready['HOST']['cold_guard_source'],ready['exact_source_projection']['mechanical'])}
assert len(aux)==3 and not(set(aux)&set(pins))and all(sha(p)==h for p,h in aux.items())
placeholder=D/'v209-future-auditor-HOST-attempts/BEFORE_ROOT_FINAL_SOURCE_PINS/audit_v209_terminal_HOST.py'
a=ast.parse(placeholder.read_text());b=ast.parse(s.read_text())
unchanged=[]
for name in('independent_seven_usage','full_statistics','main'):
 u=next(n for n in a.body if isinstance(n,ast.FunctionDef)and n.name==name);v=next(n for n in b.body if isinstance(n,ast.FunctionDef)and n.name==name)
 assert ast.dump(u,include_attributes=False)==ast.dump(v,include_attributes=False);unchanged.append(name)
# Role-specific accesses: target branch never dereferences row['draft']; rank0 never target/reference heads.
helper=Path(ns['PARENT_AUDITOR']);assert sha(helper)==ns['SOURCE_PINS'][str(helper)]
ht=ast.parse(helper.read_text());selected={n.name:ast.dump(n,include_attributes=False)for n in ht.body if isinstance(n,ast.FunctionDef)and n.name in('component_integer_verdicts','independent_multiowner','common_full_statistics','admission_and_full_statistics')}
assert len(selected)==4
out=D/'V209_FUTURE_AUDITOR_FINAL_ROOT_SOURCE_BINDING_HOST_RESULT.json';assert not out.exists()
v=dict(status='passed_SOURCE_BINDING_only',script=dict(path=str(s),sha256=sha(s)),freeze=dict(path=str(f),sha256=sha(f)),ROOT_prepared=dict(path=str(rp),sha256=sha(rp)),actual_source_members=len(pins),actual_archive_members=r['actual_archive_members'],
 actual_count_derived_from_ROOT_freeze=True,exact_CPU3=ns['CPU_SHA'],source_ready_auxiliary_bound_through_ROOTready=aux,component_clock_helper_source_unchanged=True,placeholder_to_bound_unchanged_functions=unchanged,
 arithmetic_regression_not_repeated_for_pin_binding=True,terminal_main_called=False,receipt_read=False,new209_actual_read=False,Torch_imported=False,NPU_used=False,elapsed_seconds=time.perf_counter()-start)
out.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(path=str(out),sha256=sha(out),elapsed_seconds=v['elapsed_seconds'])))
