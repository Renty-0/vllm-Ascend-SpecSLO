"""Source-only audit binding. Never traverse206 actual or any203 runtime path."""
import ast,contextlib,hashlib,io,json,pathlib,re,sys,tarfile,time,tokenize,datetime
D=pathlib.Path('/root/SpecSLO/budget-calibration-20261015/budget-diagnostics')
P=D.parent/'tree-b4-joint-subtree-single-writer-diagnostic-v206'
script=D/'audit_v206_terminal_HOST.py';old=D/'audit_v198_terminal_HOST.py'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def desc(p):return dict(path=str(p),sha256=sha(p),bytes=pathlib.Path(p).stat().st_size)
def namespace(text):
 out=[]
 for token in tokenize.generate_tokens(io.StringIO(text).readline):
  value=token.string;literal=None
  if token.type in(tokenize.NAME,tokenize.COMMENT,tokenize.STRING):
   if token.type==tokenize.STRING:
    try:literal=ast.literal_eval(value)
    except Exception:pass
   if not(isinstance(literal,str)and re.fullmatch('[0-9a-f]{64}',literal)):value=value.replace('198','206')
  out.append(token._replace(string=value))
 return tokenize.untokenize(out)
start=time.perf_counter()
assert sha(old)=='71c813a003564c60b41e16b6cf0c580e4cc907370c76960d8e658f5d8ec6fbb9'
assert sha(script)=='d2a6e8715de2c569fa2835605f40ca573d636f83218d2cf16a54b034aedbe9ea'
tree=ast.parse(script.read_text(),filename=str(script));compile(tree,str(script),'exec')
oldtree=ast.parse(namespace(old.read_text()))
functions=lambda t:{n.name:ast.dump(n,include_attributes=False)for n in t.body if isinstance(n,ast.FunctionDef)}
a,b=functions(oldtree),functions(tree)
assert set(a)==set(b)and {k for k in b if a[k]!=b[k]}=={'provenance'}
unchanged=sorted(set(a)-{'provenance'})
access=[];original_open=pathlib.Path.open
# Only pinned stdlib helper source opens occur when loading definitions/probing CLI.
def opened(p,*args,**kw):
 value=str(p)
 assert str(P/'actual')not in value and not value.startswith(str(P/'receipt-'))
 assert 'tree-empty-active-slo-admission1-causal80-active32-full-v203/actual'not in value
 assert 'tree-empty-active-slo-admission1-causal80-active32-full-v203/receipt-'not in value
 access.append(value);return original_open(p,*args,**kw)
namespace_dict={'__file__':str(script),'__name__':'_V206_SOURCE_ONLY_AUDITOR_DEFINITIONS'}
pathlib.Path.open=opened;original_argv=sys.argv;cli=[]
try:
 exec(compile(tree,str(script),'exec'),namespace_dict)
 assert namespace_dict['P']==P and namespace_dict['ACTION']=='b4-joint-subtree-single-writer-bounded-gain'
 for name,args,error in[
  ('missing_terminal_authorization',['--tag','source-only-not-authorized','--expected-receipt-sha256','0'*64,'--output',str(D/'NOT_CREATED_V206_SOURCE_ONLY.json')],SystemExit),
  ('unsafe_tag',['--terminal-authorized','--tag','../forbidden','--expected-receipt-sha256','0'*64,'--output',str(D/'NOT_CREATED_V206_SOURCE_ONLY.json')],ValueError),
  ('non_exact_receipt_sha',['--terminal-authorized','--tag','source-only-not-authorized','--expected-receipt-sha256','invalid','--output',str(D/'NOT_CREATED_V206_SOURCE_ONLY.json')],ValueError)]:
  before=len(access);sys.argv=[str(script),*args]
  try:
   with contextlib.redirect_stderr(io.StringIO()):namespace_dict['main']()
  except error as e:
   assert not isinstance(e,SystemExit)or e.code==2
   assert len(access)==before;cli.append(dict(name=name,exception=type(e).__name__,additional_file_reads=0))
  else:raise AssertionError('expected fail-closed '+name)
finally:
 pathlib.Path.open=original_open;sys.argv=original_argv
assert set(access)=={str(D/'audit_v173_terminal_full463_HOST.py'),str(D/'audit_v197_terminal_HOST.py')}
assert not any(n.split('.')[0]in namespace_dict['BACKENDS']for n in sys.modules)
# New fixed source/archive only, including retained historical pinned sources;
# no206 actual/receipt or current203 runtime read/stat.
fp=P/'FROZEN_SOURCE206.json';assert sha(fp)=='4b9019fe70d88f6ba0dc03442bbdfb6b0ecb6748b405ddf08d5c65f5fa15e2dd';f=read(fp);pins={}
for key in('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):
 for p,h in f[key].items():assert p not in pins or pins[p]==h;pins[p]=h
pins.update({str(P/'CPU_RESULT206.json'):f['CPU_sha256'],str(fp):sha(fp),f['root_review']['path']:f['root_review']['sha256']})
assert len(pins)==247 and all(sha(p)==h for p,h in pins.items())
archive=P/'PREPARED_SOURCE_ARCHIVE206.tar.gz'
assert sha(archive)=='48a97c9706ca92028bcdcc4933f464edadc801519b13e67c80f1a74781ffad1b'
with tarfile.open(archive,'r:gz')as t:
 members=t.getmembers();assert len(members)==len({m.name for m in members})==247 and {'/'+m.name for m in members}==set(pins)
 for member in members:assert member.isfile()and hashlib.sha256(t.extractfile(member).read()).hexdigest()==pins['/'+member.name]
root=D/'ROOT_V206_PREPARED_SOURCE_HOST_AUDIT.json'
assert sha(root)=='55f11c6882875deec2592a1787048313263a12d55da26c38148d406bb6a0e208'
r=read(root);assert r['status']=='passed_source_frozen_unrun'and r['actual_source_members']==r['actual_archive_members']==247
c=read(P/'CPU_RESULT206.json');prep=read(P/'PREPARED_HOST206.json')
assert c['schema']=='V206.actual-bounded-selector-metadata-bank-HOST.v1'and c['tests_run']==5 and c['status']=='passed'
assert all(c[k]is False for k in('Torch_imported','NPU_used','model_constructed','model_KV_qualified','old_suites_rerun','freeze_created','formal_Goodput'))
assert(len(prep['actual_HOST_attempts']),len(prep['retained_failed_attempt_inputs']),len(f['retained_HOST_evidence']))==(1,25,27)
base_tree=ast.parse((P/'base206.py').read_text());parent=next(n for n in base_tree.body if isinstance(n,ast.FunctionDef)and n.name=='parent_archive')
assert 'source_derivation198'in ast.unparse(parent)and 'verify_derivation198(archive=True)'in ast.unparse(parent)
result=dict(schema='V206.future-terminal-auditor-source-check.v1',status='passed_source_only',candidate=str(P),script=desc(script),check_script=desc(__file__),derived_from=desc(old),all_functions_except_provenance_namespace_AST_unchanged=unchanged,stdlib_definition_load=True,definition_source_reads=access,fail_closed_CLI=cli,source_CPU_schema=c['schema'],CPU_original_current_checks=5,retained_attempts=1,retained_files=25,retained_evidence=27,current_selected_source_SHA_checked=247,own_archive=dict(**desc(archive),exact_regular_members=247,exact_member_set_and_bytes=True),freeze=desc(fp),ROOT_prepared=desc(root),dependency_producer='base206.parent_archive original+source_derivation198320; no raw parent_ready equality',elapsed_seconds=time.perf_counter()-start,auditor_provenance_or_valid_receipt_main_executed=False,candidate_or_guard_imported=False,candidate206_actual_or_receipt_read_or_stat=False,current203_actual_or_receipt_read_or_stat=False,Torch_imported=False,NPU_used=False,model_or_device_work=False,old_suites_rerun=False,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
p=D/'V206_INDEPENDENT_TERMINAL_AUDITOR_SOURCE_CHECK.json';assert not p.exists()
with p.open('x')as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(dict(**desc(p),elapsed_seconds=result['elapsed_seconds'])))
