"""Actual pinned methods/arrival driver AST with explicit HOST-only carriers."""
import ast,copy,enum,importlib.util,json,math,sys,time,types,unittest,gzip
from pathlib import Path
from types import SimpleNamespace as NS
import publication_witness as w
HOST_ROOT=None
DETAILS={}
class Kind(enum.IntEnum):DELTA=0;CUMULATIVE=1;FINAL_ONLY=2
class Finish(enum.IntEnum):STOP=0;LENGTH=1
class Modules:
 def __enter__(self):
  self.name='vllm.sampling_params';self.old=sys.modules.get(self.name);self.mod=types.ModuleType(self.name);self.mod.RequestOutputKind=Kind;sys.modules[self.name]=self.mod
 def __exit__(self,*args):
  if self.old is None:sys.modules.pop(self.name,None)
  else:sys.modules[self.name]=self.old

def actual_def(node,path,ns):
 module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),copy.deepcopy(node)],type_ignores=[])
 exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns);return ns[node.name]
def fixture(name,cls=w.PublicationWitness):
 root=HOST_ROOT/name;root.mkdir();data=w.pins();op=Path(data['output_processor_path']);bp=Path(data['benchmark_path'])
 pns=dict(FinishReason=Finish,RequestOutputKind=Kind,OutputProcessorOutput=lambda **kw:NS(**kw))
 original=actual_def(w.method(op,'OutputProcessor','process_outputs'),op,pns)
 request_fn=actual_def(w.method(op,'RequestState','make_request_output'),op,pns)
 class Processor:
  process_outputs=original
  def __init__(self):self.request_states={};self.tracing_enabled=False;self.calls=0;self.finished=0;self.outputs=[]
  def _update_stats_from_output(self,state,out,timestamp,iteration_stats):
   if out.new_token_ids:
    if state.stats.first_token_ts is None:state.stats.first_token_ts=timestamp
    state.stats.last_token_ts=timestamp
  def _update_stats_from_finished(self,*args):pass
  def _finish_request(self,state):self.request_states.pop(state.request_id);self.finished+=1
 processor_module=types.ModuleType('HOST_original_OutputProcessor');processor_module.__file__=str(op);processor_module.OutputProcessor=Processor
 ap=bp.parent.parent/'admission.py';spec=importlib.util.spec_from_file_location('_official_HOST_admission_'+name,ap);ad=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ad;spec.loader.exec_module(ad)
 benchmark=types.ModuleType('HOST_original_official_benchmark');benchmark.__file__=str(bp);benchmark.__dict__.update(copy=copy,time=time,decide_prefill_coalesce=ad.decide_prefill_coalesce)
 fn=next(n for n in ast.parse(bp.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_run_online_arrivals');actual_def(fn,bp,vars(benchmark))
 class Engine:
  def __init__(self):self.processor=Processor();self.params=[];self.returns=[];self.packet_calls=0;self.final_outputs=[];self.error=None;self.original_calls=0
  def has_unfinished_requests(self):return bool(self.processor.request_states)
  def add_request(self,rid,prompt,params,arrival_time):
   self.params.append(params);state=NS(request_id='internal-'+rid,external_req_id=rid,parent_req=None,output_kind=params.output_kind,streaming_input=False,queue=None,
    is_prefilling=True,routed_experts_chunks=[],logprobs_processor=NS(update_from_output=lambda o:None),stream_interval=1,
    stats=NS(first_token_ts=None,last_token_ts=None,arrival_time=arrival_time),detokenizer=NS(output_token_ids=[]))
   state.detokenizer.update=lambda tokens,stop:(state.detokenizer.output_token_ids.extend(tokens)or None)
   state.make_request_output=types.MethodType(request_fn,state)
   state._new_completion_output=lambda *args:NS(token_ids=list(state.detokenizer.output_token_ids))
   state._new_request_output=lambda external,outputs,finished,kv=None,prompt_routed_experts=None:NS(request_id=external,outputs=outputs,finished=finished,metrics=state.stats)
   self.processor.request_states[state.request_id]=state
  def step(self):
   self.packet_calls+=1;outs=[]
   for internal,state in tuple(self.processor.request_states.items()):
    k=len(state.detokenizer.output_token_ids);final=k==2
    outs.append(NS(request_id=internal,new_token_ids=[100+k],pooling_output=None,finish_reason=Finish.LENGTH if final else None,
     finished=final,stop_reason=None,kv_transfer_params=None,routed_experts=None,prefill_stats=None))
   returned=self.processor.process_outputs(outs,engine_core_timestamp=time.monotonic(),iteration_stats=NS())
   self.returns.append(returned);self.final_outputs.extend(returned.request_outputs)
   return returned.request_outputs
 witness=cls(root,expected_count=2,output_tokens=3);binding=w.install(processor_module,benchmark,witness)
 engine=Engine();llm=NS(llm_engine=engine)
 rows=[dict(request_id='r'+str(i),arrival_offset_sec=0.,max_tokens=3,slo_class='tight',slo_tpot_ms=40.)for i in range(2)]
 params=[NS(output_kind=Kind.CUMULATIVE,max_tokens=3)for _ in rows]
 return root,witness,benchmark,engine,llm,rows,params,original,Processor
class Tests(unittest.TestCase):
 def test01_actual_original_driver_and_FINAL_ONLY_methods(self):
  with Modules():
   root,witness,benchmark,e,llm,rows,params,old,cls=fixture('complete')
   result=benchmark._run_online_arrivals(llm,[[1],[2]],params,rows,2,request_limit=64)
   self.assertEqual(e.processor.finished,2);self.assertEqual(e.packet_calls,3);self.assertEqual([len(v.request_outputs)for v in e.returns],[0,0,2])
   self.assertTrue(all(p.output_kind==Kind.FINAL_ONLY for p in e.params));self.assertTrue(all(p.output_kind==Kind.CUMULATIVE for p in params))
   raw=json.loads(gzip.decompress((root/'generation-000001.json.gz').read_bytes()));self.assertEqual(raw['status'],'complete');self.assertEqual(len(raw['events']),6)
   self.assertTrue(raw['evidence']['all_first_last_equal_original_engine_metrics']);self.assertTrue(raw['evidence']['all_raw_tokens_equal_original_final'])
   self.assertEqual(result[1],raw['evidence']['original_online_elapsed_seconds']);self.assertEqual(result[2],raw['clock']['mono_minus_wall']);self.assertIsNone(witness.active)
   self.assertEqual(raw['original_processor_calls'],3);self.assertTrue(raw['binding']['driver_AST_projection']['entire_original_AST_inverse'])
   DETAILS['actual_complete']=dict(original_methods=True,actual_driver=True,FINAL_ONLY_nonfinal_none_preserved=True,all_raw_tokens_and_first_last_equal=True,postlude_after_original_elapsed=True)
 def test02_exact_once_arguments_identity_inactive_and_error(self):
  with Modules():
   root,witness,benchmark,e,llm,rows,params,old,cls=fixture('identity')
   sentinel=object();calls=[];a=object();b=object()
   def original(processor,*args,**kwargs):calls.append((processor,args,kwargs));return sentinel
   self.assertIs(witness.observe(original,e.processor,(a,),dict(engine_core_timestamp=b)),sentinel)
   self.assertEqual(len(calls),1);self.assertIs(calls[0][1][0],a);self.assertIs(calls[0][2]['engine_core_timestamp'],b)
   witness.begin(rows,2);witness.bind_clock(1.,2.,1.)
   # Observer rejects invalid metadata but still calls the original once with
   # the identical argument convention and exact return object.
   self.assertIs(witness.observe(original,e.processor,(a,),dict(engine_core_timestamp=b)),sentinel);self.assertEqual(len(calls),2)
   self.assertIs(calls[-1][1][0],a);self.assertIs(calls[-1][2]['engine_core_timestamp'],b);self.assertIsNotNone(witness.active['observer_error'])
   error=ValueError('same original exception')
   def fails(*args,**kwargs):calls.append('error');raise error
   try:witness.observe(fails,e.processor,([],),dict(engine_core_timestamp=3.))
   except ValueError as actual:self.assertIs(actual,error)
   else:self.fail('original error swallowed')
   self.assertEqual(calls.count('error'),1);witness.active=None
   DETAILS['identity']=dict(inactive_and_active_original_called_once=True,original_arguments_return_and_exception_identity=True,observer_error_deferred=True)
 def test03_observer_failure_drains_original_and_retains_failure(self):
  class Failing(w.PublicationWitness):
   def snapshot(self,*args):raise OSError('HOST injected recording failure')
  with Modules():
   root,witness,benchmark,e,llm,rows,params,old,cls=fixture('observer-failure',Failing)
   with self.assertRaises(RuntimeError):benchmark._run_online_arrivals(llm,[[1],[2]],params,rows,2,request_limit=64)
   self.assertEqual(e.processor.finished,2);self.assertFalse(e.has_unfinished_requests());self.assertIsNone(witness.active)
   raw=json.loads(gzip.decompress((root/'generation-000001.json.gz').read_bytes()));self.assertEqual(raw['status'],'observer-validation-failed');self.assertIn('HOST injected',raw['observer_error'])
   self.assertTrue(all(v.output_kind==Kind.FINAL_ONLY for v in e.params))
   # The same raw artifact cannot be silently overwritten.
   witness.active=dict(generation=1,events=[],counts={},finished=set())
   with self.assertRaises(FileExistsError):witness.save('forbidden-overwrite')
   witness.active=None
   DETAILS['failure']=dict(original_all_requests_drained=True,failed_raw_exclusive=True,mode_unchanged=True)
 def test04_original_failure_and_wrong_metrics_are_not_success(self):
  with Modules():
   root,witness,benchmark,e,llm,rows,params,old,cls=fixture('original-failure');error=ValueError('original engine step failed')
   def broken():raise error
   e.step=broken
   try:benchmark._run_online_arrivals(llm,[[1],[2]],params,rows,2,request_limit=64)
   except ValueError as actual:self.assertIs(actual,error)
   else:self.fail('original exception replaced')
   raw=json.loads(gzip.decompress((root/'generation-000001.json.gz').read_bytes()));self.assertEqual(raw['status'],'original-failed');self.assertIsNone(witness.active)
   root,witness,benchmark,e,llm,rows,params,old,cls=fixture('bad-metrics')
   original_step=e.step
   def corrupt_metrics():
    value=original_step()
    for output in value:output.metrics.last_token_ts+=1.
    return value
   e.step=corrupt_metrics
   with self.assertRaises(RuntimeError):benchmark._run_online_arrivals(llm,[[1],[2]],params,rows,2,request_limit=64)
   raw=json.loads(gzip.decompress((root/'generation-000001.json.gz').read_bytes()));self.assertEqual(raw['status'],'observer-validation-failed');self.assertIn('first/last',raw['error']['message'])
   DETAILS['negative']=dict(original_error_preserved=True,raw_vs_original_metrics_mismatch_rejected=True,failed_not_reclassified=True)
