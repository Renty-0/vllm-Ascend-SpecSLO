"""Stdlib-only official TP4 engine-publication observer; imports install nothing."""
from __future__ import annotations
import ast,copy,functools,gzip,hashlib,inspect,json,math,textwrap,time
from pathlib import Path
P=Path(__file__).resolve().parent

def require(ok,message):
 if not ok:raise RuntimeError('official publication witness: '+message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pins():
 data=json.loads((P/'SOURCE_PINS.json').read_text())
 for path,h in data['files'].items():require(sha(path)==h,'exact readonly source '+path)
 return data
def digest(node):return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
def method(path,owner,name):
 unit=ast.parse(Path(path).read_text());cls=next(n for n in unit.body if isinstance(n,ast.ClassDef)and n.name==owner)
 return next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name==name)
def checked_callable(fn,node,path):
 path=Path(path)
 require(Path(fn.__code__.co_filename).resolve()==path.resolve(),'actual callable filename '+node.name)
 # Parse the same whole-line segment as inspect.getsource. Dedenting only
 # one side would change multiline docstring whitespace in a class method.
 lines=path.read_text().splitlines(keepends=True)[node.lineno-1:node.end_lineno]
 expected=ast.parse(textwrap.dedent(''.join(lines))).body[0]
 actual=ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
 require(digest(actual)==digest(expected),'actual original callable AST '+node.name)

def run_ast(source):
 unit=ast.parse(source);old=next(n for n in unit.body if isinstance(n,ast.FunctionDef)and n.name=='_run_online_arrivals');new=copy.deepcopy(old)
 matches=[i for i,n in enumerate(new.body)if isinstance(n,ast.Assign)and ast.unparse(n)=='mono_minus_wall = mono_start - wall_start']
 require(len(matches)==1,'one exact original arrival clock bridge')
 at=matches[0]+1;extra=ast.parse('_publication_witness.bind_clock(wall_start,mono_start,mono_minus_wall)').body[0];new.body.insert(at,extra)
 reverse=copy.deepcopy(new);reverse.body.pop(at);require(digest(reverse)==digest(old),'whole original arrival driver inverse')
 return new,dict(original_AST_sha256=digest(old),projected_AST_sha256=digest(new),one_inserted_clock_record_after_original_samples=True,entire_original_AST_inverse=True)

class PublicationWitness:
 def __init__(self,root,*,expected_count=463,output_tokens=256):
  self.root=Path(root);require(self.root.is_absolute(),'private absolute output directory')
  self.expected_count=expected_count;self.output_tokens=output_tokens;self.ordinal=0;self.active=None;self.binding=None
 def begin(self,rows,count):
  require(self.active is None,'nonoverlapping original online generations')
  self.ordinal+=1
  self.active=dict(generation=self.ordinal,source_rows=[dict(v)for v in rows[:count]],count=count,
   events=[],packets=0,counts={},finished=set(),observer_error=None,clock=None,
   wrapper_start_perf=time.perf_counter(),wrapper_start_wall=time.time(),original_processor_calls=0)
  if count!=self.expected_count:self.mark('unexpected original request count')
 def mark(self,message):
  if self.active is not None and self.active['observer_error']is None:
   self.active['observer_error']=str(message)
 def bind_clock(self,wall_start,mono_start,mono_minus_wall):
  # This is the only new statement inside the original arrival driver. Do not
  # fail before its model/drain path: qualification errors surface afterwards.
  try:
   require(self.active is not None and self.active['clock']is None,'one actual original clock bridge')
   self.active['clock']=dict(wall_start=wall_start,mono_start=mono_start,mono_minus_wall=mono_minus_wall)
  except BaseException as error:self.mark(type(error).__name__+': '+str(error))
 def snapshot(self,processor,outputs,timestamp):
  a=self.active;require(a is not None,'active generation')
  require(type(outputs)is list and type(timestamp)in(int,float)and math.isfinite(timestamp)and timestamp>0,'actual original engine output list/timestamp')
  packet=a['packets'];a['packets']+=1;rows=[]
  for out in outputs:
   internal=out.request_id;state=processor.request_states.get(internal)
   require(state is not None,'raw request has actual pre-processing state')
   external=state.external_req_id;tokens=list(out.new_token_ids)
   require(type(external)is str and all(type(t)is int and t>=0 for t in tokens),'raw external ID and Python token IDs')
   require(out.pooling_output is None and state.parent_req is None and not state.streaming_input,'target-only one-output nonstreaming-input scope')
   require(external not in a['finished'],'no post-finish token event')
   start=a['counts'].get(external,0);a['counts'][external]=start+len(tokens)
   final=bool(out.finished);reason=out.finish_reason
   if final:a['finished'].add(external)
   rows.append(dict(packet=packet,internal_request_id=internal,request_id=external,
    engine_core_timestamp=timestamp,token_offset=start,token_ids=tokens,finished=final,
    finish_reason=None if reason is None else int(reason),original_output_kind=str(state.output_kind),
    original_processor_returned=False))
  return rows
 def observe(self,original,processor,args,kwargs):
  if self.active is None:return original(processor,*args,**kwargs)
  a=self.active;pending=[]
  try:
   outputs=args[0]if args else kwargs['engine_core_outputs']
   timestamp=args[1]if len(args)>1 else kwargs.get('engine_core_timestamp')
   pending=self.snapshot(processor,outputs,timestamp)
  except BaseException as error:self.mark(type(error).__name__+': '+str(error))
  # Exactly once, original object identities and caller argument convention.
  try:result=original(processor,*args,**kwargs)
  except BaseException as error:
   self.mark('original process_outputs raised '+type(error).__name__)
   try:a['events'].extend(pending)
   except BaseException:pass
   raise
  a['original_processor_calls']+=1
  try:
   for row in pending:row['original_processor_returned']=True
   a['events'].extend(pending)
  except BaseException as error:self.mark(type(error).__name__+': '+str(error))
  return result
 def validate(self,result):
  outputs,elapsed,bridge,timing=result;a=self.active
  require(a is not None and a['observer_error']is None,'observer source/recording error '+str(None if a is None else a['observer_error']))
  require(a['clock']is not None and bridge==a['clock']['mono_minus_wall']and math.isfinite(elapsed)and elapsed>0,'exact original returned clock bridge and elapsed')
  source={str(v['request_id']):v for v in a['source_rows']}
  require(len(source)==a['count']==self.expected_count and len(outputs)==self.expected_count,'all original output requests')
  events={key:[]for key in source}
  for row in a['events']:
   require(row['request_id']in events and row['original_processor_returned']is True,'actual successful original processor event identity')
   events[row['request_id']].append(row)
  evidence=[];seen=set()
  for output in outputs:
   key=str(output.request_id);require(key in source and key not in seen,'unique original final output ID');seen.add(key)
   actual=list(output.outputs[0].token_ids);saved=events[key];tokens=[];first=None;last=None;final=0;previous=-1.
   for e in saved:
    require(e['token_offset']==len(tokens)and e['engine_core_timestamp']>=previous,'contiguous offsets and ordered original publication timestamps')
    previous=e['engine_core_timestamp'];tokens.extend(e['token_ids']);final+=int(e['finished'])
    if e['token_ids']:
     if first is None:first=e['engine_core_timestamp']
     last=e['engine_core_timestamp']
   require(tokens==actual and len(tokens)==self.output_tokens and final==1 and saved[-1]['finished'],'exact full raw event-to-final256 token reconstruction')
   metrics=output.metrics
   require(first==metrics.first_token_ts and last==metrics.last_token_ts,'raw first/last exactly equal original engine metrics')
   require(a['clock']['mono_start']<=first<=last<=a['clock']['mono_start']+elapsed+1e-6,'actual publication within original online generation clock')
   evidence.append(dict(request_id=key,output_tokens=len(tokens),output_token_ids_sha256=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),
    first_engine_timestamp=first,last_engine_timestamp=last,original_metrics_first=metrics.first_token_ts,original_metrics_last=metrics.last_token_ts,
    paper_tpot_ms=(last-first)*1000/len(tokens),observed_tpot_ms=(last-first)*1000/max(1,len(tokens)-1),
    planned_arrival_wall=a['clock']['wall_start']+float(source[key].get('arrival_offset_sec',0.)),
    slo_class=source[key].get('slo_class'),slo_tpot_ms=source[key].get('slo_tpot_ms')))
  require(seen==set(source)==set(a['counts'])==a['finished'],'all original requests complete exactly once')
  return dict(all_raw_tokens_equal_original_final=True,all_first_last_equal_original_engine_metrics=True,
   original_online_elapsed_seconds=elapsed,original_online_timing=timing,request_evidence=evidence)
 def save(self,status,*,evidence=None,error=None):
  a=self.active;record={k:v for k,v in a.items()if k not in('counts','finished')}
  record.update(schema='official-TP4.raw-engine-publication-witness.v1',status=status,evidence=evidence,error=error,
   counts=a['counts'],finished=sorted(a['finished']),binding=self.binding,
   event_clock='original EngineCoreOutputs.timestamp, not observer receive time or client delivery',
   original_FINAL_ONLY_unchanged=True,original_model_sampling_scheduling_arrivals_accounting_unchanged=True,
   per_token_device_or_clock_calls=False,original_clock_sampled_before_log=True,
   publication_metric_is_not_client_delivery=True,device_qualification=False)
  raw=json.dumps(record,allow_nan=False,separators=(',',':')).encode();path=self.root/f'generation-{a["generation"]:06d}.json.gz'
  with path.open('xb')as stream:stream.write(gzip.compress(raw,mtime=0))
  return path
 def run(self,original,*args,**kwargs):
  rows=args[3]if len(args)>3 else kwargs['request_rows'];count=args[4]if len(args)>4 else kwargs['count']
  self.begin(rows,count)
  try:
   try:result=original(*args,**kwargs)
   except BaseException as error:
    self.active['wrapper_finish_perf']=time.perf_counter()
    try:self.save('original-failed',error=dict(type=type(error).__name__,message=str(error)))
    except BaseException as retain:error.add_note('Publication partial retention failed: '+str(retain))
    raise
   self.active['wrapper_finish_perf']=time.perf_counter()
   # Original elapsed/bridge/online_timing have already been sampled. Strict
   # observer validation and gzip are postlude, outside that original wall.
   try:evidence=self.validate(result)
   except BaseException as error:
    self.save('observer-validation-failed',error=dict(type=type(error).__name__,message=str(error)));raise
   self.save('complete',evidence=evidence)
   return result
  finally:self.active=None

def install(processor_module,benchmark,witness):
 data=pins();processor_path=Path(data['output_processor_path']);benchmark_path=Path(data['benchmark_path'])
 require(Path(processor_module.__file__).resolve()==processor_path and Path(benchmark.__file__).resolve()==benchmark_path,'exact original loaded modules')
 cls=processor_module.OutputProcessor;old=cls.process_outputs;old_run=benchmark._run_online_arrivals
 checked_callable(old,method(processor_path,'OutputProcessor','process_outputs'),processor_path)
 node=next(n for n in ast.parse(benchmark_path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='_run_online_arrivals');checked_callable(old_run,node,benchmark_path)
 require(not getattr(old,'_official_publication_observer',False),'install once')
 projected,proof=run_ast(benchmark_path.read_text());ns=dict(vars(benchmark),_publication_witness=witness)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),projected],type_ignores=[])),str(P/'<original-online-arrival-clock-record>'),'exec'),ns)
 delegated=ns['_run_online_arrivals']
 @functools.wraps(old)
 def process(processor,*args,**kwargs):return witness.observe(old,processor,args,kwargs)
 process._official_publication_observer=True
 @functools.wraps(old_run)
 def online(*args,**kwargs):return witness.run(delegated,*args,**kwargs)
 cls.process_outputs=process;benchmark._run_online_arrivals=online
 witness.binding=dict(output_processor_sha256=data['files'][str(processor_path)],benchmark_sha256=data['files'][str(benchmark_path)],driver_AST_projection=proof,
  original_processor_called_once=True,original_call_arguments_and_return_identity_preserved=True,
  active_only_original_online_driver=True,original_FINAL_ONLY_unchanged=True)
 return dict(witness.binding)
