"""Source-only proof. No candidate imports, actual/receipt access, or model work."""
import ast,contextlib,datetime,hashlib,io,json,pathlib,sys,tarfile,time
D=pathlib.Path('/root/SpecSLO/budget-calibration-20261015/budget-diagnostics')
P=D.parent/'tree-empty-active-slo-admission1-causal80-active32-full-v203'
s=D/'audit_v203_terminal_HOST.py';old=D/'audit_v197_terminal_HOST.py'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def desc(p):return dict(path=str(p),sha256=sha(p),bytes=pathlib.Path(p).stat().st_size)
start=time.perf_counter()
assert sha(s)=='f7e12023dc4aafb31628871ae987378476e20042ea3bb5be633c6266e6cc5ab4'
assert sha(old)=='5118f3e4f2da7ad3de169dfce9e48f310df8097c41c7e0b350df8852eedc81b7'
tree=ast.parse(s.read_text(),filename=str(s));compile(tree,str(s),'exec')
original=ast.parse(old.read_text().replace('197','203'))
def functions(t):return {n.name:ast.dump(n,include_attributes=False)for n in t.body if isinstance(n,ast.FunctionDef)}
a,b=functions(original),functions(tree)
unchanged=['component_integer_verdicts','common_full_statistics','role_counter_fields','independent_multiowner']
assert all(a[k]==b[k]for k in unchanged)
assert {k for k in b if a.get(k)!=b[k]}=={'provenance','admission_and_full_statistics','main'}
reads=[];original_open=pathlib.Path.open
# Only original173 stdlib helper definitions may be read during definition/CLI probes.
def opened(p,*args,**kw):
 text=str(p)
 assert '/actual/'not in text and '/receipt-'not in text,('forbidden mutable output access',text)
 reads.append(text)
 return original_open(p,*args,**kw)
namespace={'__name__':'_V203_SOURCE_ONLY_AUDITOR_DEFINITIONS','__file__':str(s)}
pathlib.Path.open=opened;cli=[];original_argv=sys.argv
try:
 exec(compile(tree,str(s),'exec'),namespace)
 assert namespace['P']==P and namespace['ACTION']=='empty-active-slo-admission1-causal80-active32-full463'
 for name,args,expected in [
  ('missing_terminal_authorization',['--tag','source-only-never-authorized','--expected-receipt-sha256','0'*64,'--output',str(D/'NOT_CREATED_V203_SOURCE_ONLY.json')],SystemExit),
  ('unsafe_tag',['--terminal-authorized','--tag','../forbidden','--expected-receipt-sha256','0'*64,'--output',str(D/'NOT_CREATED_V203_SOURCE_ONLY.json')],ValueError),
  ('non_exact_receipt_sha',['--terminal-authorized','--tag','source-only-never-authorized','--expected-receipt-sha256','invalid','--output',str(D/'NOT_CREATED_V203_SOURCE_ONLY.json')],ValueError)]:
  before=len(reads);sys.argv=[str(s),*args]
  try:
   with contextlib.redirect_stderr(io.StringIO()):namespace['main']()
  except expected as e:
   assert not isinstance(e,SystemExit)or e.code==2
   assert len(reads)==before
   cli.append(dict(name=name,exception=type(e).__name__,additional_file_reads=0))
  else:raise AssertionError('expected fail-closed CLI '+name)
finally:
 pathlib.Path.open=original_open;sys.argv=original_argv
assert set(reads)=={str(D/'audit_v173_terminal_full463_HOST.py')}
assert not any(n.split('.')[0]in namespace['BACKENDS']for n in sys.modules)
# Check the already-frozen source archive, not runtime output or model qualification.
fpath=P/'FROZEN_SOURCE203.json';f=read(fpath)
assert sha(fpath)=='7ff2a3d5c3d6b7bca985bf843bb46b29b71d1cb05428ccf2b17ba86741893d75'
pins={}
for key in('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):
 for path,h in f[key].items():
  assert path not in pins or pins[path]==h
  pins[path]=h
pins.update({str(P/'CPU_RESULT203.json'):f['CPU_sha256'],str(fpath):sha(fpath),f['root_review']['path']:f['root_review']['sha256']})
assert len(pins)==332 and all(sha(p)==h for p,h in pins.items())
archive=P/'PREPARED_SOURCE_ARCHIVE203.tar.gz'
assert sha(archive)=='1f39cfc60c6c0d64c464c33fa146fb9ac1d91b080c6d07c9fec4aaa59002ce0a'
with tarfile.open(archive,'r:gz')as t:
 members=t.getmembers()
 assert len(members)==len({m.name for m in members})==332 and {'/'+m.name for m in members}==set(pins)
 for m in members:
  assert m.isfile()and hashlib.sha256(t.extractfile(m).read()).hexdigest()==pins['/'+m.name]
c=read(P/'CPU_RESULT203.json')
assert c['schema']=='V203.actual-empty-active-admission-HOST.v1'and c['tests_run']==3 and c['status']=='passed'
assert c['local_before']==c['local_after']==f['files']and c['external_before']==c['external_after']==f['external_files']
assert all(c[k]is False for k in('Torch_imported','NPU_used','model_constructed','parent_suites_rerun','device_qualification_inherited','freeze_created'))
prepared=read(P/'PREPARED_HOST203.json')
assert(len(prepared['actual_HOST_attempts']),len(prepared['retained_HOST_files']),len(f['retained_HOST_evidence']))==(3,84,88)
cp=ast.parse((P/'check203.py').read_text());parent=next(n for n in cp.body if isinstance(n,ast.FunctionDef)and n.name=='require_parent')
assert 'inherited_dependencies'in ast.unparse(parent)and 'members=len(expected)'in ast.unparse(parent)
ap=ast.parse((P/'admission203.py').read_text());apfn=next(n for n in ap.body if isinstance(n,ast.FunctionDef)and n.name=='audit_trace')
assert 'spec_rhythm_slo_aware_admission_enabled'in ast.unparse(apfn)and 'spec_rhythm_prefill_coalesce_deferred_polls'in ast.unparse(apfn)
record=dict(schema='V203.independent-future-terminal-auditor-source-check.v1',status='passed_source_only',candidate=str(P),script=desc(s),check_script=desc(__file__),derived_from=desc(old),source_AST_parsed_and_compiled=True,unchanged_original197_function_AST=unchanged,changed_functions=['provenance','admission_and_full_statistics','main'],stdlib_definition_load=True,definition_reads=reads,fail_closed_CLI=cli,current_selected_sources_checked=332,own_archive=dict(**desc(archive),regular_members_checked=332,exact_member_set=True,exact_member_bytes=True),source_CPU_schema=c['schema'],CPU_tests_run=3,all3_attempts_and84_retained_files_source_bound=True,source_artifacts=[desc(P/x)for x in('CPU_RESULT203.json','PREPARED_HOST203.json','FROZEN_SOURCE203.json','check203.py','admission203.py','empty203.py','audit203.py')],root_prepared=desc(D/'ROOT_V203_PREPARED_SOURCE_HOST_AUDIT.json'),prior_source_check=desc(D/'V203_AUDITOR_SOURCE_CHECK_ATTEMPT1_NOTE.json'),elapsed_seconds=time.perf_counter()-start,auditor_main_valid_receipt_executed=False,actual_or_receipt_read_or_stat=False,candidate_or_guard_imported=False,Torch_imported=False,NPU_used=False,models_launched=False,device_queries=False,old_suites_rerun=False,source_qualification_inherited=False,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
out=D/'V203_INDEPENDENT_TERMINAL_AUDITOR_SOURCE_CHECK.json'
assert not out.exists()
with out.open('x')as stream:json.dump(record,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(dict(**desc(out),elapsed_seconds=record['elapsed_seconds'])))
