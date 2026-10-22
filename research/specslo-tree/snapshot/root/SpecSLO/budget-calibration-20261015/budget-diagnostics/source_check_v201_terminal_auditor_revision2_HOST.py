"""REV2 auditor source/provenance-format check only; no auditor main execution."""
import ast,copy,datetime,hashlib,json,pathlib,sys,time
D=pathlib.Path('/root/SpecSLO/budget-calibration-20261015/budget-diagnostics')
P=D.parent/'tree-normal-b4-final-kv5-cohort16-full-v201'
old=D/'audit_v201_terminal_HOST.py';new=D/'audit_v201_terminal_HOST_revision2.py'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def desc(p):return dict(path=str(p),sha256=sha(p),bytes=pathlib.Path(p).stat().st_size)
start=time.perf_counter()
assert sha(old)=='678b9a359a69f87f5745544db86c6b89206bbd54def94736f410589cb328dbc4'
assert sha(new)=='1c3eb7aafb01e346c43d9d76e750e7448ce910a69a52ef3595a479ff7da56729'
a="parent==readiness['parent_ready']"
b="json.loads(json.dumps(parent,sort_keys=True,allow_nan=False))==readiness['parent_ready']"
assert old.read_text().count(a)==1 and new.read_text().count(b)==1
assert new.read_text().replace(b,a)==old.read_text()
oldtree=ast.parse(old.read_text());tree=ast.parse(new.read_text());compile(tree,str(new),'exec')
oldfunc={n.name:ast.dump(n,include_attributes=False)for n in oldtree.body if isinstance(n,ast.FunctionDef)}
newfunc={n.name:ast.dump(n,include_attributes=False)for n in tree.body if isinstance(n,ast.FunctionDef)}
assert {k for k in newfunc if oldfunc.get(k)!=newfunc[k]}=={'provenance'}
manifest=D/'v201-terminal-auditor-REV1-before-json-representation-fix/PRESERVED_MANIFEST.json'
assert sha(manifest)=='e9c7bdb4a4c20cc1f9dcf881779f3cc6074762a33e7ccd63413d43f906994081'
kept=read(manifest)
assert len(kept['files'])==5
for p,row in kept['files'].items():assert sha(p)==sha(row['snapshot'])==row['sha256']
receipt_path=P/'receipt-final-kv5-v201-full463-v1.json'
assert sha(receipt_path)=='27afc66978e620322af1c6cf209bbe64df816383ad9e4b70dc9ff22f2c6b5d78'
receipt=read(receipt_path);assert receipt['status']=='failed'
# Load only pure auditor definitions; no main/provenance/full-only audit called.
namespace={'__name__':'_V201_REVISION2_SOURCE_ONLY_CHECK','__file__':str(new)}
exec(compile(tree,str(new),'exec'),namespace)
sys.meta_path.insert(0,namespace['Firewall']());sys.path.insert(0,str(P))
import base201
parent=base201.parent_ready()
ready=receipt['readiness'];normalized=json.loads(json.dumps(parent,sort_keys=True,allow_nan=False))
assert parent!=ready['parent_ready']and normalized==ready['parent_ready']
provenance=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='provenance')
checks=[n for n in ast.walk(provenance)if isinstance(n,ast.Compare)and isinstance(n.left,ast.Call)and ast.unparse(n.left)=="json.loads(json.dumps(parent, sort_keys=True, allow_nan=False))"]
assert len(checks)==1
condition=compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(checks[0]))),str(new)+'::actual-normalization-expression','eval')
def equal(live,saved):return eval(condition,dict(json=json,parent=live,readiness=dict(parent_ready=saved)))
assert equal(parent,ready['parent_ready'])
negative=[]
saved=ready['parent_ready']
for name,mutate in[
 ('missing_identity',lambda x:x.pop('identity')),
 ('extra_field',lambda x:x.update(unexpected_source_override=True)),
 ('wrong_identity_sha',lambda x:x['identity'].__setitem__('freeze_sha256','0'*64)),
 ('wrong_mapper_partition_value',lambda x:x['component_actual_evidence']['mapper_partitions'][0].__setitem__(0,-100)),
 ('missing_mapper_partition',lambda x:x['component_actual_evidence']['mapper_partitions'].pop()),
 ('changed_formal_flag',lambda x:x.__setitem__('formal_Goodput',True))]:
 value=copy.deepcopy(saved);mutate(value);assert not equal(parent,value);negative.append(name)
assert not namespace['Firewall'].attempts and not any(k.split('.')[0]in namespace['BACKENDS']for k in sys.modules)
proof=read(D/'V201_INDEPENDENT_TERMINAL_AUDITOR_PARENT_READY_SCHEMA_DIFF.json')
assert proof['dependency_archives_raw_equality_passed']is True and len(proof['type_differences'])==6
out=dict(schema='V201.terminal-auditor-REVISION2-source-check.v1',status='passed_source_only_pending_ROOT_review_before_execution',candidate=str(P),script=desc(new),old_script=desc(old),check_script=desc(__file__),preserved_REV1=desc(manifest),schema_diff=desc(D/'V201_INDEPENDENT_TERMINAL_AUDITOR_PARENT_READY_SCHEMA_DIFF.json'),authorized_failed_receipt=desc(receipt_path),one_expression_inverse_exact=True,all_other_functions_AST_unchanged=True,all_source_archive_hash_permit_checker_gates_unchanged=True,actual_parent_ready_producer=True,actual_normalization_expression_executed=True,parent_live_tuple_to_saved_list_difference_count=6,all_other_fields_equal_after_JSON_roundtrip=True,negative_field_value_tests=negative,dependency_archive_equality_source_proof_unchanged=True,revision2_auditor_main_executed=False,revision2_provenance_or_terminal_audit_executed=False,old_auditor_main_rerun=False,models_launched=False,Torch_imported=False,NPU_used=False,device_queries=False,old_suites_rerun=False,frozen_candidate_edited=False,elapsed_seconds=time.perf_counter()-start,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
p=D/'V201_TERMINAL_HOST_AUDITOR_SOURCE_CHECK_REVISION2.json';assert not p.exists()
with p.open('x')as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(**desc(p),elapsed_seconds=out['elapsed_seconds'])))
