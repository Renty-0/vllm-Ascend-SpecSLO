"""Direct ROOT-owned full runner; one constructor and all selected windows."""
import argparse,gzip,json,os,re,sys,traceback
from functools import partial
from pathlib import Path
from base190 import P,BASE,ACTION,PARENT_ACTION,require,parent_ready

def save(directory,name,value):
 path=Path(directory)/name
 raw=json.dumps(value,allow_nan=False,separators=(',',':')).encode()
 if name.endswith('.gz'):raw=gzip.compress(raw,mtime=0)
 with path.open('xb') as f:f.write(raw)

def worker(cfg,rank,port,connection,admission_connection=None,compact_worker_result_metrics=False,
           *,diagnostic_spec,own_identity,output_directory,admission_windows,extension_module='extension190'):
 engine=None
 try:
  from check190 import require_ready
  require_ready(own_identity)
  from diagnostic_authorization import activate_local_diagnostic
  with activate_local_diagnostic(diagnostic_spec):
   import importlib
   extension=importlib.import_module(extension_module)
   require(extension_module=='extension190','this fixed runner uses only its pinned three callbacks')
   entry,ControlsClass=extension.prepare_install(output_directory)
   from config_snapshot_v138 import snapshot_config
   config_before=snapshot_config(cfg)
   import torch
   os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),RANK=str(rank),WORLD_SIZE='4',LOCAL_RANK=str(rank))
   if os.environ.get('HCCL_NPU_SOCKET_PORT_RANGE')=='auto':os.environ.pop('HCCL_NPU_SOCKET_PORT_RANGE')
   from vllm_ascend.spec_decode.pearl import native_engine as native,native_graph as graph
   from original_tree_cycle_v75 import capture_original_protocol_before_startup
   protocol=capture_original_protocol_before_startup(native)
   policy=entry['install'](fast=True,host=True,transport=True,replay=True,repair=True,prefill=True)
   import tail_hidden_observer,live_decode_patch
   tail_hidden_observer.install(graph)
   overlay=extension.after_install(native,graph,live_decode_patch)
   from cpu_default_device_cleanup_host import capture_before_native_constructor
   state=capture_before_native_constructor(torch)
   origins_before=extension.loaded_origins()
   engine=native.NativePearlEngine(cfg)
   engine._v101_output_directory=output_directory
   origins_after=extension.loaded_origins()
   from vllm_ascend.spec_decode.pearl import spec_rhythm as rhythm
   combination=json.loads((BASE/'CANDIDATE_V132.json').read_text())
   controls=ControlsClass(native=native,rhythm=rhythm,policy=policy,torch=torch,
    constructor_state=state,is_draft=engine.is_draft,configuration=combination['controls'])
   engine._v75_original_protocol=protocol
   connection.send(('ready',rank,engine.graph_metrics()))
   while True:
    mode,action,_,_=connection.recv()
    if mode=='exit':break
    require(mode=='calibrate' and action==ACTION,'fixed current full action')
    from startup_receipts_v75 import prepare_original_draft_family,stage4,NATIVE_COUNTERS,FAILURE_COUNTERS
    from source190 import startup_auditor
    stage=lambda label,callback:stage4(engine,'V190/'+label,callback)
    with torch.inference_mode():
     prepare_original_draft_family(engine)
     startup=startup_auditor(output_directory)(engine)
     engine._v190_actual_startup=startup
     engine._v173_actual_startup=startup
     report=dict(schema='V190.active-window-full-worker.v1',rank=rank,action=ACTION,errors=[],
      startup=startup,startup_completed=True,configuration_before_constructor=config_before,
      configuration_after_constructor=snapshot_config(engine.config),
      actual_helper_origins_before_constructor=origins_before,actual_helper_origins_after_constructor=origins_after,
      private_IO=dict(output=str(output_directory),read_inputs=str(BASE),parent_directory_written=False,
       entry_AST_source=str(P/'source190.py'),inventory_AST_source=str(P/'progress190.py')),
      overlay=overlay,old_21scope26phase_executed=False,formal_Goodput=False,adopted=False)
     stage('candidate-component-controls',lambda:controls.select('candidate',engine=engine))
     component=extension.qualify_component(engine,native,stage,policy,output_directory)
     report['component']=component
     from scope190 import issue,report as scope_report
     scope=stage('issue-current-standalone-scope',lambda:issue(engine,component_proof=component))
     report['standalone_scope']=stage('current-standalone-scope',lambda:scope_report(scope,engine))
     from trace190 import run_windows,warm_to_fixedpoint
     report['prewarm']=warm_to_fixedpoint(engine,scope=scope,stage=stage,output=output_directory,
      entry=entry,admission_windows=admission_windows)
     def checkpoint(traces,metrics):
      path=Path(output_directory)/f'rank{rank}-window-checkpoint.json.gz';temp=path.with_suffix('.tmp')
      temp.write_bytes(gzip.compress(json.dumps(dict(traces=traces,metrics=metrics),allow_nan=False).encode(),mtime=0));temp.replace(path)
     report['windows']=run_windows(engine,controls=controls,scope=scope,stage=stage,
      directory=Path(output_directory)/'windows',entry=entry,output=output_directory,
      admission_windows=admission_windows,checkpoint=checkpoint)
     report['sealed']=bool(engine.graph_runner.graph_cache_sealed)
     report['full463_completed']=True
     report['final_controls']=stage('final-original-controls',controls.observe)
     report['Native_final']=stage('final-Native',lambda:{k:engine.graph_metrics()[k] for k in NATIVE_COUNTERS})
     require(not any(report['Native_final'][k] for k in FAILURE_COUNTERS),'no Native failure/fallback final')
     stage('final-release',engine._release_cache)
     report['cache_released']=engine.cache_allocation is None and engine.cache_block_tables is None
     save(output_directory,f'rank{rank}-local-diagnostic.json',report)
    connection.send(('calibrated',rank,dict(windows=list(admission_windows),full463=True)))
 except BaseException:
  error=traceback.format_exc()
  try:save(output_directory,f'rank{rank}-last-error.json',dict(rank=rank,error=error,formal_Goodput=False,partial_causal80=getattr(engine,'_v190_causal80_partial',{})))
  except BaseException:error+='\nretention failed\n'+traceback.format_exc()
  try:connection.send(('error',error))
  except BaseException:pass
 finally:
  connection.close()
  if admission_connection is not None:admission_connection.close()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--action',choices=(ACTION,),required=True)
 parser.add_argument('--tag',required=True);parser.add_argument('--windows',default='32')
 args=parser.parse_args();require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.tag) is not None,'fresh safe tag')
 from window160 import parse_windows
 windows=parse_windows(args.windows);require(windows==(32,),'only active32 in this new arm')
 from check190 import require_ready
 readiness=require_ready();spec=dict(action=PARENT_ACTION,identity=readiness['parent']['identity'])
 output=P/'actual'/args.tag;require(not output.exists(),'fresh actual output');output.mkdir(parents=True,exist_ok=False)
 from diagnostic_authorization import activate_local_diagnostic
 with activate_local_diagnostic(spec):
  from config190 import make_config
  from vllm_ascend.spec_decode.pearl import api
  api._pearl_worker=partial(worker,diagnostic_spec=spec,own_identity=readiness['identity'],
   output_directory=str(output),admission_windows=windows)
  engine=None
  try:
   engine=api.PEARLEngine(make_config());engine._send_all(('calibrate',ACTION,None,None))
   replies=engine._receive_all('V190 '+ACTION)
   require(sorted(replies,key=lambda row:row[1])==[('calibrated',i,dict(windows=list(windows),full463=True)) for i in range(4)],'actual four full workers')
   from audit190 import audit
   result=audit(output,windows)
   save(output,'completed.json',dict(action=ACTION,replies=replies,audit=result,formal_Goodput=False,adopted=False))
  finally:
   if engine is not None:engine.exit()
   save(output,'worker-exit.json',dict(closed=bool(engine and engine._closed),all_exited=bool(engine and all(not p.is_alive() for p in engine._processes))))
if __name__=='__main__':main()
