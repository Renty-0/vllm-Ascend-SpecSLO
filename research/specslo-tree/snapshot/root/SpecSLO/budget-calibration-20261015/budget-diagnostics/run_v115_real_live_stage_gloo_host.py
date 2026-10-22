"""Uninstalled HOST-only real frozen Live.stage AST compact-transport equivalence."""
import ast
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

P=Path(__file__).resolve().parent
FROZEN=P.parent/'tree-s35-production-v101'
TAG='V115_REAL_LIVE_STAGE_GLOO_HOST_20261016_v1'
OUT=P/TAG
CASES=('success_ascii','success_utf8_63bytes','success_utf8_64bytes','callback_ValueError',
 'callback_KeyboardInterrupt','name_mismatch','sequence_mismatch','rank_mismatch',
 'long_name','invalid_surrogate','tuple_name','huge_sequence','negative_sequence','bool_rank')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_original_and_candidate():
    source=FROZEN/'live_native_boundaries.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LiveNativeBoundaries')
    original=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='stage')
    original=copy.deepcopy(original)
    native=ast.parse((FROZEN/'native_boundaries.py').read_text())
    rejected=next(n for n in native.body if isinstance(n,ast.ClassDef) and n.name=='BoundaryRejected')
    namespace={'__name__':'_real_frozen_live_stage_AST'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[rejected,original],type_ignores=[])),str(source),'exec'),namespace)
    old=namespace['stage']
    candidate=copy.deepcopy(original)
    calls=[(i,n) for i,n in enumerate(candidate.body) if isinstance(n,ast.Expr)
           and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='dist.all_gather_object']
    assert len(calls)==1
    i,node=calls[0];call=node.value
    assert len(call.args)==2 and ast.unparse(call.args[0])=='rows'
    assert len(call.keywords)==1 and call.keywords[0].arg=='group' and ast.unparse(call.keywords[0].value)=='self.group'
    assert isinstance(call.args[1],ast.Call) and ast.unparse(call.args[1].func)=='dict'
    candidate.body[i]=ast.Assign(targets=[ast.Name(id='rows',ctx=ast.Store())],
      value=ast.Call(func=ast.Attribute(value=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='transport',ctx=ast.Load()),attr='collect',ctx=ast.Load()),
      args=[copy.deepcopy(call.args[1])],keywords=[]))
    candidate=ast.fix_missing_locations(candidate)
    # Verify the only AST mutation is the one approved status-transport statement.
    restored=copy.deepcopy(candidate);restored.body[i]=copy.deepcopy(node)
    assert ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False)
    exec(compile(ast.Module(body=[candidate],type_ignores=[]),str(source)+':one-status-transport-delta','exec'),namespace)
    return old,namespace['stage'],namespace['BoundaryRejected'],dict(
       source=str(source),source_sha256=sha(source),BoundaryRejected_source_sha256=sha(FROZEN/'native_boundaries.py'),
       original_AST_sha256=hashlib.sha256(ast.dump(original,include_attributes=False).encode()).hexdigest(),
       single_delta='dist.all_gather_object(rows, original_row, group=self.group) -> rows=self.transport.collect(original_row)',
       all_other_AST_identical=True)

def worker(rank,rendezvous,cores):
    os.sched_setaffinity(0,{cores[rank%len(cores)]})
    import torch
    import torch.distributed as dist
    from compact_host_stage_transport import CompactHostStageTransport
    assert 'torch_npu' not in sys.modules
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    dist.init_process_group('gloo',rank=rank,world_size=4,init_method='file://'+rendezvous,
       timeout=datetime.timedelta(seconds=60))
    old,new,Rejected,provenance=load_original_and_candidate()
    records=[]
    try:
        for cpu_context in (False,True):
            torch.set_default_device('cpu' if cpu_context else None)
            for inference in (False,True):
                transport=CompactHostStageTransport(torch,dist,dist.group.WORLD)
                with torch.inference_mode(inference):
                    for case in CASES:
                        stage='live-prefix-write-declaration';sequence=71;reported_rank=rank
                        if case=='success_utf8_63bytes':stage='汉'*21
                        if case=='success_utf8_64bytes':stage='é'*32
                        if case=='name_mismatch' and rank==1:stage='different-stage'
                        if case=='sequence_mismatch' and rank==3:sequence+=1
                        if case=='rank_mismatch' and rank==3:reported_rank=2
                        if case=='long_name':stage='x'*65
                        if case=='invalid_surrogate':stage='bad\ud800'
                        if case=='tuple_name':stage=('original','tuple')
                        if case=='huge_sequence':sequence=2**80
                        if case=='negative_sequence':sequence=-1
                        if case=='bool_rank' and rank==1:reported_rank=True
                        results=[]
                        for method in (old,new):
                            boundary=SimpleNamespace(engine=SimpleNamespace(rank=reported_rank),
                               group=dist.group.WORLD,sequence=sequence,events=[],transport=transport)
                            value={'rank':rank,'returned':'actual-callback-value'};count=[0]
                            def callback():
                                count[0]+=1
                                if case=='callback_ValueError' and rank==2:raise ValueError('actual expected failure: exact message')
                                if case=='callback_KeyboardInterrupt' and rank==0:raise KeyboardInterrupt('actual expected interrupt: exact message')
                                return value
                            exception=None;returned=None
                            try:
                                returned=method(boundary,stage,callback)
                            except BaseException as exc:
                                exception=exc
                            assert count[0]==1 and boundary.sequence==sequence+1 and len(boundary.events)==1
                            if exception is None:assert returned is value
                            else:assert type(exception) is Rejected
                            results.append(dict(returned_value=returned,returned_same_callback_object=returned is value,
                              callback_count=count[0],sequence=boundary.sequence,events=boundary.events,
                              exception_class=None if exception is None else type(exception).__name__,
                              exception_text=None if exception is None else str(exception)))
                        assert results[0]==results[1],(rank,cpu_context,inference,case,results)
                        event=results[1]['events'][0]
                        assert event['boundary_kind']=='host-status-agreement'
                        records.append(dict(CPU_DeviceContext=cpu_context,inference_mode=inference,case=case,
                           exact_return_callback_sequence_events_exception=True,evidence=results[1]))
                    send,receive=transport.buffers()
                    assert not torch.is_inference(send) and not torch.is_inference(receive)
                    # Source is created inside the actual outer inference mode;
                    # the retained transport copies it into non-inference buffers.
                    source=torch.tensor([1],dtype=torch.int64,device='cpu')
                    assert torch.is_inference(source)==inference
                records.append(dict(CPU_DeviceContext=cpu_context,inference_mode=inference,
                    tensor_buffer_normal=True,source_inference_tensor=inference,
                    compact_calls=transport.compact_calls,object_fallbacks=transport.object_fallbacks))
            torch.set_default_device(None)
        report=dict(rank=rank,passed=True,records=records,AST_provenance=provenance,
          torch_version=torch.__version__,CPU_affinity=sorted(os.sched_getaffinity(0)),
          torch_npu_imported='torch_npu' in sys.modules,model_initialized=False,NPU_initialized=False,
          production_installed=False)
        with (OUT/f'rank{rank}.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    finally:
        torch.set_default_device(None)
        dist.destroy_process_group()

def main():
    assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0'
    assert sys.version_info[:2]==(3,11)
    OUT.mkdir(exist_ok=False)
    paths=[Path(__file__),P/'compact_host_stage_transport.py',P/'compact_host_stage_codec.py',
      FROZEN/'live_native_boundaries.py',FROZEN/'native_boundaries.py']
    before={str(p):sha(p) for p in paths}
    load_original_and_candidate()
    import torch.multiprocessing as mp
    start=time.perf_counter()
    try:
        mp.spawn(worker,args=(str(OUT/'rendezvous'),sorted(os.sched_getaffinity(0))[:4]),nprocs=4,join=True)
    except BaseException as exc:
        report=dict(schema='HOST.V115.real-frozen-Live-stage-AST-Gloo-equivalence.v1',passed=False,
          error=type(exc).__name__+': '+str(exc),source_before=before,actual_HOST_argv=sys.argv,
          source_after={str(p):sha(p) for p in paths},elapsed_seconds=time.perf_counter()-start,
          NPU_started=False,production_installed=False)
        with (P/(TAG+'_FAILED.json')).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
        raise
    reports=[json.loads((OUT/f'rank{r}.json').read_text()) for r in range(4)]
    after={str(p):sha(p) for p in paths}
    assert before==after and all(r['passed'] and not r['torch_npu_imported'] for r in reports)
    report=dict(schema='HOST.V115.real-frozen-Live-stage-AST-Gloo-equivalence.v1',passed=True,
      actual_HOST_argv=sys.argv,source_before=before,source_after=after,
      actual_CPU_Gloo_ranks=4,CPU_default_modes=['None','CPU_DeviceContext'],inference_modes=[False,True],
      cases_per_mode_per_rank=len(CASES),cases=list(CASES),
      total_old_new_stage_comparisons=4*4*len(CASES),
      exact_comparison=['callback returned object','callback invocation count','sequence','full events/rank rows',
        'BoundaryRejected exact class','exception str byte-for-byte'],
      buffer_proof='non-inference CPU send/receive allocated under inference_mode(False); actual source follows outer mode',
      provenance=reports[0]['AST_provenance'],rank_artifacts={str(OUT/f'rank{r}.json'):sha(OUT/f'rank{r}.json') for r in range(4)},
      elapsed_seconds=time.perf_counter()-start,NPU_started=False,model_initialized=False,production_installed=False,
      Native_chunk_commit_fences_changed=False,whole_workload_gain_proved=False,formal_Goodput=False)
    with (P/(TAG+'.json')).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,path=str(P/(TAG+'.json')),sha256=sha(P/(TAG+'.json')),
      comparisons=report['total_old_new_stage_comparisons'],elapsed_seconds=report['elapsed_seconds'])))
if __name__=='__main__':main()
