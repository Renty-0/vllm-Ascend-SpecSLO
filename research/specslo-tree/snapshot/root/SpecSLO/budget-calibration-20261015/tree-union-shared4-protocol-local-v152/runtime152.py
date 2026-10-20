"""Explicit future device entry; imports and HOST checks initialize no backend."""
import argparse,gzip,json,os,re,sys,traceback
from functools import partial
from pathlib import Path
P=Path(__file__).resolve().parent
if str(P) in sys.path:sys.path.remove(str(P))
sys.path.insert(0,str(P))
from scope152 import ACTION,require
import reuse152

def save(directory,name,value):
 raw=json.dumps(value,allow_nan=False,separators=(',',':')).encode()
 if name.endswith('.gz'):raw=gzip.compress(raw,mtime=0)
 with (Path(directory)/name).open('xb') as stream:stream.write(raw)
 require(sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file())<32*1024**2,'private32MiB runtime report cap')
def install(native,graph):
 from reuse152 import install_runtime_helpers
 origins=install_runtime_helpers(native,graph)
 import hidden151,observer151
 hidden151.install(native);observer151.install(native,graph)
 record=graph.NativeACLGraphRunner._record_execution
 def tracked(runner,kind,execution,output,*,entry_key=None,token_rows=()):
  if kind=='target':runner._v151_last_target_key=entry_key
  return record(runner,kind,execution,output,entry_key=entry_key,token_rows=token_rows)
 graph.NativeACLGraphRunner._record_execution=tracked
 return origins

def worker(cfg,rank,port,connection,admission_connection=None,compact_worker_result_metrics=False,*,identity,directory):
 engine=None;events=[]
 try:
  from check152 import require_ready
  require_ready(identity)
  import torch
  os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),RANK=str(rank),WORLD_SIZE='4',LOCAL_RANK=str(rank))
  if os.environ.get('HCCL_NPU_SOCKET_PORT_RANGE')=='auto':os.environ.pop('HCCL_NPU_SOCKET_PORT_RANGE')
  from vllm_ascend.spec_decode.pearl import native_engine as native,native_graph
  origins=install(native,native_graph);engine=native.NativePearlEngine(cfg)
  if not engine.is_draft:engine.graph_runner._dynamic_tree_target_replay_first=True
  connection.send(('ready',rank,engine.graph_metrics()))
  while True:
   command,action,_,_=connection.recv()
   if command=='exit':break
   require(command=='calibrate' and action==ACTION,'exact local issued action')
   from actor152 import run_actor,native as snapshot
   from boundary151 import stage4
   stage=lambda n,f:stage4(engine,events,'V152/'+n,f)
   with torch.inference_mode():
    def constructor():
     runner=engine.graph_runner
     require(not runner.entries and not runner.draft_entries and not runner.target_entries and not any(snapshot(engine).values())
      and engine.config.enable_spec_rhythm is engine.config.precompile_decode_graphs is engine.config.precompile_serial_draft_graphs is False
      and engine.cache_allocation is None and engine.cache_block_tables is None,'own empty Native constructor; no unrelated inventories')
     return dict(actual_origins=origins,Native_constructor_calls=1,storage_rows=engine._cache_storage_sequence_capacity,
      max_num_seqs=engine.config.max_num_seqs,max_num_queued_seqs=engine.config.max_num_queued_seqs,pages=engine.config.num_kvcache_blocks)
    construction=stage('own-constructor',constructor);result=run_actor(engine,native,stage)
    stage('final-cache-release',engine._release_cache)
    report=dict(action=ACTION,rank=rank,role='draft' if engine.is_draft else 'target',construction=construction,result=result,
     boundary_events=events,errors=[],cache_released=engine.cache_allocation is None and engine.cache_block_tables is None,
     Native_flags_assigned=False,formal_Goodput=False)
    save(directory,f'rank{rank}-component.json.gz',report)
   connection.send(('calibrated',rank,dict(cases=4,complete_zeros=2,timings=0)))
 except BaseException:
  error=traceback.format_exc()
  try:save(directory,f'rank{rank}-last-error.json',dict(rank=rank,error=error,boundary_events=events,partial_actor=getattr(engine,'_v151_partial',None),qualification=False,formal_Goodput=False))
  except BaseException:pass
  try:connection.send(('error',error))
  except BaseException:pass
 finally:
  if engine is not None and (engine.cache_allocation is not None or engine.cache_block_tables is not None):
   try:engine._release_cache()
   except BaseException:pass
  connection.close()
  if admission_connection is not None:admission_connection.close()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--action',choices=(ACTION,),required=True);parser.add_argument('--tag',required=True)
 args=parser.parse_args();require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.tag) is not None,'safe private unique tag')
 from check152 import require_ready
 ready=require_ready();directory=P/'actual'/args.tag;directory.mkdir(parents=True,exist_ok=False)
 from vllm_ascend.spec_decode.pearl import api
 from scope152 import make_config
 cfg=make_config();save(directory,'prerequisites.json',ready)
 api._pearl_worker=partial(worker,identity=ready['identity'],directory=str(directory));engine=None
 try:
  engine=api.PEARLEngine(cfg);engine._send_all(('calibrate',ACTION,None,None));replies=engine._receive_all('V152 parent3-normal4-shared4 local observed protocol')
  require(sorted(replies,key=lambda r:r[1])==[('calibrated',i,dict(cases=4,complete_zeros=2,timings=0)) for i in range(4)],'all4 completed actual scopes')
  from audit152 import audit
  reports=[json.loads(gzip.decompress((directory/f'rank{r}-component.json.gz').read_bytes())) for r in range(4)]
  save(directory,'completed.json',dict(action=ACTION,replies=replies,audit=audit(reports),formal_Goodput=False))
 finally:
  if engine is not None:engine.exit()
  save(directory,'worker-exit.json',dict(closed=bool(engine and engine._closed),all_exited=bool(engine and all(not p.is_alive() for p in engine._processes))))
if __name__=='__main__':main()
