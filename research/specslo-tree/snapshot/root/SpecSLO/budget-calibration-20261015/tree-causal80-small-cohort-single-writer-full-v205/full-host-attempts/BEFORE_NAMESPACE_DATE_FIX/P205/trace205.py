"""P160 actual trace with private shell identities and standalone-only scope."""
import ast,copy,gzip,json,math,statistics
from pathlib import Path
from base205 import P,BASE,ACTIVE,checked,require,functions,inherited_module,sha
from source205 import original_observation
from scope205 import report as standalone_report
from window160 import adapt_source,window
from probe160 import trace_source
FILENAME='<V205-original-live-with-active-window>'

def make_trace(active_window,*,entry,output):
 active_window=window(active_window)
 live=inherited_module('live_decode_patch');observation=inherited_module('live_poisson_probe')
 original=inherited_module('production_probe_v101');proxy=original_observation(entry,output)
 namespace=dict(vars(live));state={}
 functions(BASE/'live_decode_patch.py',namespace,names=('install',),changes=((
  '<V81-bounded-live-mixed-prefill>',FILENAME,1),))
 installer=namespace['install']
 def install(native,factory):
  builder=entry.get('_live_builder',live.build_live_runtime)
  require(callable(builder),'explicit current live source builder')
  from empty205 import patch
  state['source']=patch(adapt_source(builder(),active_window))
  namespace['build_live_runtime']=lambda:state['source']
  return installer(native,factory)
 assertion_ns=dict(vars(observation),original=proxy,LIVE_FILENAME=FILENAME)
 def installed_source():
  require('source' in state,'actual cap source installed before its assertion');return state['source']
 assertion_ns['_live_source']=installed_source
 functions(BASE/'live_poisson_probe.py',assertion_ns,names=('_assert_live_decode',))
 ns=dict(vars(original),original=proxy,install=install,_assert_live_decode=assertion_ns['_assert_live_decode'],standalone_report=standalone_report)
 tree=trace_source(active_window);changed=0
 for node in ast.walk(tree):
  if isinstance(node,ast.keyword) and node.arg=='S35_structural_scope':
   require(ast.unparse(node.value)=="__import__('shape_scope_v101').scope_report(qualified_s35_scope, engine)",'original trace scope-report expression')
   node.arg='standalone_scope205';node.value=ast.parse('standalone_report(qualified_s35_scope,engine)',mode='eval').body;changed+=1
 require(changed==1,'exact one standalone scope-report replacement')
 exec(compile(ast.fix_missing_locations(tree),str(P/'trace205.py')+'::actual-trace','exec'),ns)
 return ns['run_trace'],state

def metrics(info):
 ns=dict(globals())
 functions(ACTIVE/'metrics160.py',ns,names=('_percentiles','full_metrics'))
 return ns['full_metrics'](info)

def run_windows(engine,*,controls,scope,stage,directory,entry,output,admission_windows,checkpoint=None):
 ns=dict(globals(),require_ready=__import__('check205').require_ready,
         make_trace=lambda w:make_trace(w,entry=entry,output=output),full_metrics=metrics)
 from window160 import DEFAULT_WINDOWS,windows
 ns.update(DEFAULT_WINDOWS=DEFAULT_WINDOWS,windows=windows)
 functions(ACTIVE/'pair160.py',ns,names=('run_pair',))
 return ns['run_pair'](engine,controls=controls,scope=scope,stage=stage,directory=directory,
  admission_windows=admission_windows,checkpoint=checkpoint)

def warm_to_fixedpoint(engine,*,scope,stage,output,entry,admission_windows,max_attempts=8):
 """Real counter convergence before the unchanged P160 warm/zero schedule."""
 from candidate_phases_v132 import split_snapshot
 import torch.distributed as dist
 records=[]
 for cap in admission_windows:
  trace,_=make_trace(cap,entry=entry,output=output)
  converged=False
  for attempt in range(max_attempts):
   label='V205/prewarm/window'+str(cap)+'/attempt'+str(attempt)
   lane=lambda name,fn:stage(label+'/'+name,fn)
   folder=Path(output)/'prewarm'/('window'+str(cap))/str(attempt)
   lane('fresh-output',lambda:folder.mkdir(parents=True,exist_ok=False)
        if engine.rank==engine.topology.draft_leader_rank else None)
   before=lane('admission-reset',lambda:split_snapshot(engine,reset_normal_admission=True))
   value=trace(engine,count=65,role='warmup65',directory=folder,qualified_s35_scope=scope)
   def agree():
    packet=dict(rank=engine.rank,native=value['native'],source_generation=value['source_generation'])
    rows=[None]*4;dist.all_gather_object(rows,packet,group=engine.groups.verification_coordination_group)
    require({r['rank'] for r in rows}==set(range(4)) and len({r['source_generation'] for r in rows})==1,
            'common actual warm generation and ranks')
    stable=all(not any(r['native']['delta'].values()) for r in rows)
    return dict(actual_rank_records=rows,all21_all4_zero=stable)
   agreement=lane('actual21-all4-convergence',agree)
   records.append(dict(window=cap,attempt=attempt,trace=value,original_reset=before,agreement=agreement))
   if agreement['all21_all4_zero']:
    converged=True;break
  lane('bounded-convergence-gate',lambda:require(converged,'max8 real warm traces did not converge'))
 return dict(max_attempts_per_window=max_attempts,traces=records,all_windows_converged=True,
  all_extra_warm_outside_measurement=True,Native_flags_assigned=False)
