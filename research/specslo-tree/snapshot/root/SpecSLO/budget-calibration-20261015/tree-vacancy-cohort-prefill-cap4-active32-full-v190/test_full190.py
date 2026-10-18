"""Four bounded real-source HOST contracts, never model numerical evidence."""
import ast,copy,dataclasses,dis,gzip,importlib.util,json,math,sys,tempfile,time,types,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from base190 import P,B,P173,BASE,sha,read,functions,checked,core_module,parent_module
DETAILS={}
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
def definitions(path,names,namespace):
 unit=ast.parse(checked(path).read_text());nodes=[copy.deepcopy(v) for v in unit.body if isinstance(v,(ast.ClassDef,ast.FunctionDef)) and v.name in names]
 assert {v.name for v in nodes}==set(names)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])),str(path)+'::actual-HOST-definitions','exec'),namespace)
 return namespace
class Tests(unittest.TestCase):
 def test01_actual_config_builder_control_source_binding_and_inverse(self):
  import check190,config190,trace190,extension190
  source_patch190=core_module('source_patch185')
  self.assertFalse(check190.require_parent()['device_qualification_inherited'])
  # Actual new config function calls an explicit HOST constructor boundary;
  # only the two reviewed config fields can change.
  module=types.ModuleType('startup_config_v132');fields=dict(spec_rhythm_prefill_coalesce_min_requests=1,spec_rhythm_prefill_coalesce_max_wait_ms=0.,max_num_seqs=64,max_num_queued_seqs=463,num_kvcache_blocks=544)
  Config=dataclasses.make_dataclass('Config',[(k,type(v)) for k,v in fields.items()]);original=Config(**fields);module.make_config=lambda:original
  previous=sys.modules.get(module.__name__);sys.modules[module.__name__]=module
  try:value=config190.make_config()
  finally:
   if previous is None:sys.modules.pop(module.__name__)
   else:sys.modules[module.__name__]=previous
  self.assertEqual({k:(getattr(original,k),getattr(value,k)) for k in fields if getattr(original,k)!=getattr(value,k)},dict(spec_rhythm_prefill_coalesce_min_requests=(1,4),spec_rhythm_prefill_coalesce_max_wait_ms=(0.,150.)))
  from host_source_patch185 import actual_builder
  original_source=actual_builder()
  with tempfile.TemporaryDirectory(dir=P,prefix='HOST-entry190-') as out:
   entry,Controls=extension190.prepare_install(out)
   expected=parent_module('mapper173').controls_class()
   import marshal
   methods={k:v for k,v in vars(Controls).items() if isinstance(v,types.FunctionType)}
   self.assertTrue(methods)
   for name,method in methods.items():self.assertEqual(marshal.dumps(method.__code__),marshal.dumps(getattr(expected,name).__code__))
   self.assertIs(Controls.__init__.__globals__['dispatch'],parent_module('mapper173').dispatch)
   fn,state=trace190.make_trace(32,entry=entry,output=out)
   # Real install closure prepares the actual new source, then the immutable
   # installer correctly rejects this unauthorized HOST/no-model call.
   actual_trace=next(c.cell_contents for c in fn.__closure__ if isinstance(c.cell_contents,types.FunctionType))
   installer=actual_trace.__globals__['install']
   with self.assertRaises(Exception):installer(NS(),lambda e:None)
   self.assertTrue(source_patch190.assert_inverse(original_source,state['source']))
   self.assertEqual(source_patch190.unpatch(state['source']),ast.unparse(ast.parse(original_source))+'\n')
  runtime=ast.parse((P/'runtime190.py').read_text());main=next(v for v in runtime.body if isinstance(v,ast.FunctionDef) and v.name=='main')
  self.assertEqual(sum(isinstance(v,ast.Call) and ast.unparse(v.func)=='capture_original_protocol_before_startup' for v in ast.walk(runtime)),1)
  self.assertTrue(any(isinstance(v,ast.ImportFrom) and v.module=='config190' for v in ast.walk(main)))
  self.assertIs(trace190.metrics.__globals__['sha'],sha)
  DETAILS['config_source']=dict(actual_config_only_4_150=True,actual_original_control_class=True,actual_dynamic_source_inverse=True,no_model_or_permission=True)
 def test02_real_NativeCache_bank_allpages_and_original_publication(self):
  from test_host180 import engine,torch_carrier
  from prefill190 import Bank,packed,capture_begin
  rows=[dict(request_id=str(i),token_ids=[100+i]*n,length=n) for i,n in enumerate((59,130,257,385,173))]
  for n in (2,4):
   with torch_carrier():
    e=engine();bank=Bank(e,rows,n);lease=bank.guard()
    self.assertEqual(lease['candidate_owners'],list(range(n)));self.assertEqual(lease['reference_owners'],list(range(n,2*n)))
    for ref in (False,True):
     tokens,ids,positions,last=packed(bank,reference=ref)
     self.assertEqual(len(tokens),sum(len(r['token_ids']) for r in rows[:n]));self.assertEqual(len(last),n)
     for owner,pos in zip(ids,positions,strict=True):self.assertGreaterEqual(e.cache_allocation.block_tables[owner][pos//128],0)
    self.assertEqual(e.cache_allocation.num_cached_tokens[:2*n],[0]*(2*n))
    bank.guard();e._release_cache()
  # Execute actual Native dataclass and callback method, not logical mirrors.
  ns=dict(dataclass=dataclasses.dataclass,replace=dataclasses.replace,math=math,time=time)
  definitions(NATIVE/'native_engine.py',{'PearlPipelineState','NativeSamplingParams','_truncate_completion','_finished'},ns)
  module=types.ModuleType('vllm_ascend.spec_decode.pearl.spec_rhythm');nsstate=definitions(NATIVE/'spec_rhythm.py',{'SpecRhythmRuntimeState','_clamp'},dict(dataclass=dataclasses.dataclass))
  module.SpecRhythmRuntimeState=nsstate['SpecRhythmRuntimeState'];old=sys.modules.get(module.__name__);sys.modules[module.__name__]=module
  unit=ast.parse(checked(NATIVE/'native_engine.py').read_text());cls=next(v for v in unit.body if isinstance(v,ast.ClassDef) and v.name=='NativePearlEngine')
  callback=copy.deepcopy(next(v for v in cls.body if isinstance(v,ast.FunctionDef) and v.name=='_deliver_committed_tokens'))
  exec(compile(ast.fix_missing_locations(ast.Module(body=[callback],type_ignores=[])),'actualNativeCallback190','exec'),ns)
  import publication190
  try:
   for rank in range(4):
    e=NS(is_draft=rank==0,rank=rank,topology=NS(target_leader_rank=1),eos_token_ids=frozenset())
    e._deliver_committed_tokens=types.MethodType(ns['_deliver_committed_tokens'],e)
    bank=NS(n=4,prompts=[r['token_ids'] for r in rows[:4]])
    proof=publication190.publish(e,NS(**{k:ns[k] for k in ('PearlPipelineState','NativeSamplingParams','_finished')}),bank,[501,502,503,504])
    self.assertEqual(len(proof['callbacks']),4 if rank==1 else 0);self.assertEqual(bank.contexts,bank.reference_contexts)
    self.assertEqual([v[-1] for v in bank.contexts],[501,502,503,504]);self.assertNotIn('_token_commit_callback',vars(e))
  finally:
   if old is None:sys.modules.pop(module.__name__)
   else:sys.modules[module.__name__]=old
  model=NS(forward=lambda *a:'original');e=NS(model=model);before=model.forward;saved,restore=capture_begin(e)
  self.assertIsNot(model.forward,before);restore();self.assertIs(model.forward,before)
  import subprocess,os
  child=subprocess.run([sys.executable,'-B',str(P/'host_next4_child190.py')],env={**os.environ,'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},capture_output=True,text=True,timeout=90)
  self.assertEqual(child.returncode,0,child.stdout+'\n'+child.stderr)
  DETAILS['actual_next4_control']=json.loads(child.stdout.strip().splitlines()[-1])
  DETAILS['bank_publication']=dict(actual_NativePrefixCache=True,all_queried_pages_independent=True,actual_Native_logical_classes_and_callback=True,all4_publication_executed=True,capture_restore_identity=True,model_numerical_qualification=False)
 def test03_actual_closed_admission_replay_and_real_metrics_fixture(self):
  VacancyCohort=core_module('cohort185').VacancyCohort
  from trace_admission190 import audit_record
  p=VacancyCohort();p.begin(1)
  common=dict(remaining_count=30,request_limit=32,now=100.,ready_arrival_times=[1.],active_request_count=31,ready_ids=[40])
  p.decide(**common);common['now']=100.2;p.decide(**common)
  common.update(now=101.,ready_ids=list(range(50,58)),ready_arrival_times=[1.]*8,active_request_count=24)
  p.decide(**common);p.close()
  record=dict(generation=1,events=copy.deepcopy(p.events),status='complete',partial=False,closed=True,pending_episode=None,episode_count=p.episode_count)
  value=audit_record(record);self.assertEqual(value['cohort_size_histogram'],{'1':1,'8':1})
  bad=copy.deepcopy(record);bad['events'][0]['waited_ms']=150.
  with self.assertRaises(RuntimeError):audit_record(bad)
  import trace190
  from base190 import ACTIVE
  fixtures=sorted(k for k in read(P/'INPUTS190.json')['files'] if 'tree-active-window-full-v166/actual/' in k and k.endswith('full463-actual-trace.json.gz'))
  self.assertEqual(len(fixtures),4);results=[]
  for path in fixtures:
   root=B/'tree-active-window-full-v166';ns=dict(vars(trace190),P=root)
   functions(ACTIVE/'metrics160.py',ns,names=('_percentiles','full_metrics'))
   info=dict(path=path,sha256=sha(path));results.append(ns['full_metrics'](info))
   if results[-1] is not None:
    self.assertEqual(results[-1]['Native_paper']['attained'],191);self.assertAlmostEqual(results[-1]['Native_paper']['Goodput'],244.0908068587041)
    self.assertAlmostEqual(results[-1]['raw_tokens_per_second'],591.6965632229319)
   badns=dict(ns);badns.pop('sha')
   # Real function compiled from same original producer, with the old missing
   # global, reproduces its actual NameError before any file rewrite.
   oldfn=types.FunctionType(ns['full_metrics'].__code__,badns)
   with self.assertRaises(NameError):oldfn(info)
  self.assertEqual(sum(v is not None for v in results),1)
  DETAILS['trace_metrics']=dict(actual_four_completed166_artifacts=True,original_GP=244.0908068587041,old_missing_sha_NameError=True,original_clock_formula_unchanged=True,minimum4_releases_actual8=True)
 def test04_actual_guard_CLI_rows_and_safe_storage_race(self):
  import runtime190,base190
  G=B/'budget-diagnostics/v190-root-execution';q=G/'guard190.py';spec=importlib.util.spec_from_file_location('_V190_HOST_guard',q);guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
  self.assertEqual((guard.MAX_SECONDS,guard.MAX_TAG_BYTES,guard.MAX_AGGREGATE_BYTES),(7200,256*1024**2,512*1024**2))
  for path,name,target,extra in [(q,'prepare_only','f',dict(state=dict(local={},external={},evidence={},parent_archive={}),root={},peers={})),(checked(B/'budget-diagnostics/v154-root-execution/supervise_v154.py'),'execute_locked','row',dict(tag='HOST190',readiness={}))]:
   fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name);node=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==target for t in n.targets))
   ns={**vars(guard),**extra,'sha':lambda p:'0'*64 if Path(p)==guard.CPU else sha(p)}
   exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)],type_ignores=[])),'actual190-guardrow','exec'),ns)
   self.assertEqual(ns[target]['vacancy_min_requests'],4);self.assertFalse(ns[target]['formal_Goodput'])
   bad=copy.deepcopy(node);bad.value.keywords.insert(0,ast.keyword(arg='vacancy_min_requests',value=ast.Constant(4)))
   with self.assertRaises(TypeError):exec(compile(ast.fix_missing_locations(ast.Module(body=[bad],type_ignores=[])),'actual190-badrow','exec'),dict(ns))
  fn=next(n for n in ast.parse((P/'runtime190.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main');nodes=[]
  for node in fn.body:
   if isinstance(node,ast.ImportFrom) and node.module=='check190':break
   nodes.append(copy.deepcopy(node))
  code=compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'actual190-CLI','exec');old=sys.argv
  try:
   sys.argv=['runtime190.py','--action',base190.ACTION,'--tag','HOST190'];ns=dict(vars(runtime190));exec(code,ns);self.assertEqual(ns['windows'],(32,))
   sys.argv+=['--windows','16']
   with self.assertRaises(RuntimeError):exec(code,dict(vars(runtime190)))
  finally:sys.argv=old
  for shell in ('launch190.sh','startup190.sh'):self.assertIn(base190.ACTION,(G/shell).read_text())
  from unittest.mock import patch
  actual=P/'actual';actual.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=actual,prefix='HOST-race190-') as temp:
   root=Path(temp);kept=root/'kept.json';kept.write_bytes(b'123');tmp=root/'writing.tmp';tmp.write_bytes(b'x');original=Path.stat
   def race(p,*a,**kw):
    if p==tmp and kw.get('follow_symlinks',True):
     try:p.unlink()
     except FileNotFoundError:pass
    return original(p,*a,**kw)
   with patch.object(Path,'stat',race):self.assertEqual(guard.bytes_under(root),3)
   link=root/'bad.tmp';link.symlink_to(kept)
   with self.assertRaises(RuntimeError):guard.bytes_under(root)
  DETAILS['guard']=dict(actual_prepare_and_receipt_constructors=True,duplicates_rejected=True,actual_CLI32_only=True,deterministic_atomic_tmp_race=True,symlink_rejected=True)
