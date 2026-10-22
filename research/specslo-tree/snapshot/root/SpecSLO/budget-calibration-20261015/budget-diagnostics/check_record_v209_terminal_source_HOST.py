"""New recorder source-only checks; no current/future actual or receipt access."""
import ast,copy,hashlib,json,statistics,subprocess,time,traceback
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace as NS
D=Path('/root/SpecSLO/budget-calibration-20261015/budget-diagnostics')
P=D.parent/'tree-causal80-small-cohort-role-aware-audit-full-v209'
SCRIPT=D/'record_v209_terminal_result_ROOT.py'
PARENT=D/'record_v205_terminal_result_ROOT.py'
PROJECTION=D/'RECORD_V209_TERMINAL_SOURCE_PROJECTION.json'
PYTHON='/root/miniconda3/envs/vllm-hust-dev/bin/python3.11'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
 if v is not True:raise RuntimeError(m)
def run():
 start=time.perf_counter();folder=D/'record209-source-HOST-attempts'/str(time.time_ns());folder.mkdir(parents=True)
 paths=[SCRIPT,PARENT,PROJECTION,Path(__file__)]
 before={str(p):sha(p)for p in paths}
 for i,p in enumerate(paths):(folder/(str(i)+'-'+p.name)).write_bytes(p.read_bytes())
 result=dict(schema='ROOT.V209.future-terminal-recorder-source-HOST.v1',status='failed',SOURCE_ONLY=True,
  new209_actual_or_receipt_read=False,old_suites_rerun=False,Torch_imported=False,NPU_used=False,
  source_before=before)
 try:
  tree=ast.parse(SCRIPT.read_text());old=ast.parse(PARENT.read_text());projection=json.loads(PROJECTION.read_text())
  reverse=SCRIPT.read_text()
  for item in reversed(projection['changes']):
   require(reverse.count(item['after'])==item['count'],'exact source reverse uniqueness')
   reverse=reverse.replace(item['after'],item['before'])
  require(reverse==PARENT.read_text(),'whole recorder projection exactly reverses to parent205')
  funcs=lambda tree:{n.name:ast.dump(n,include_attributes=False)for n in tree.body if isinstance(n,ast.FunctionDef)}
  require(funcs(tree)==funcs(old),'all original summary/helper function AST unchanged')
  help_result=subprocess.run([PYTHON,'-B',str(SCRIPT),'--help'],capture_output=True,text=True)
  require(help_result.returncode==0 and '--terminal-authorized'in help_result.stdout and '--session-id'in help_result.stdout,'actual argparse help, no terminal read')
  noauth=subprocess.run([PYTHON,'-B',str(SCRIPT),'--session-id','82303','--expected-terminal-receipt-sha256','0'*64],capture_output=True,text=True)
  require(noauth.returncode==2 and '--terminal-authorized'in noauth.stderr,'missing explicit terminal authorization rejects in parser')
  wrongsession=subprocess.run([PYTHON,'-B',str(SCRIPT),'--terminal-authorized','--session-id','82304','--expected-terminal-receipt-sha256','0'*64],capture_output=True,text=True)
  require(wrongsession.returncode!=0 and 'AssertionError'in wrongsession.stderr and 'FileNotFoundError'not in wrongsession.stderr,'wrong session rejects before any source/terminal artifact read')
  # Execute the real result constructor against an explicitly synthetic HOST
  # schema carrier. pin/sha are pure stubs, so no future artifact is touched.
  row=next(copy.deepcopy(n)for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='result'for t in n.targets))
  constants={}
  for n in tree.body:
   if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='SOURCE_PINS':constants=ast.literal_eval(n.value)
  by_class={n:dict(requests=count,attained=1,TPOT_median_ms=10,TPOT_P90_ms=20,TTFT_median_ms=30,TTFT_P90_ms=40)for n,count in(('tight',271),('normal',96),('loose',96))}
  audit=dict(seven_usage={'HOST_carrier':True},old_padding_behavior_equal=False,numerical_variant_only=True,
   current_multiowner_eager_component={'HOST_carrier':True},native_zero_subtractions=252)
  receipt=dict(elapsed_seconds=1000,finished_UTC='HOST-only',actual_full_audit=audit,pre_verifier={'HOST_carrier':True},post_verifier={'HOST_carrier':True})
  calls=[dict(N=16,real_query_count=80,family='causal',physical_M=80)]
  ns=dict(datetime=datetime,timezone=timezone,P=P,D=D,TAG='small-tail-role-aware-v209-full463-v1',
   args=NS(session_id=82303),receipt=P/'unused-HOST-receipt.json',r=receipt,A=P/'unused-HOST-actual',raw=P/'unused-HOST-raw.gz',
   pin=lambda p:dict(path=str(p),sha256='0'*64),sha=lambda p:'0'*64,duration=200.,good=25600,attained=100,
   by_class=by_class,tpots=[10.,20.],ttft=[30.,40.],statistics=statistics,
   pct=lambda values,q:values[0]+(values[-1]-values[0])*q,original_clock_summary={'HOST_original_clock_carrier':True},
   SOURCE_PINS=constants,x=dict(active_peak=32,decode_counters=dict(spec_rhythm_prefill_batches=15,spec_rhythm_prefill_requests=463,spec_rhythm_online_refill_ms=12345)),
   calls=calls,owners=16,N=Counter({16:1}),per_rank={},hashlib=hashlib,json=json,rows=[{'completion_token_ids':[1,2]}],
   cohorts=dict(cohorts=[dict(owners=[0,1])]),Counter=Counter)
  exec(compile(ast.fix_missing_locations(ast.Module(body=[row],type_ignores=[])),str(SCRIPT)+'::actual-result-constructor-HOST','exec'),ns)
  actual=ns['result']
  require(actual['session_id']==82303 and actual['tag']=='small-tail-role-aware-v209-full463-v1'and actual['candidate']==str(P),'actual current209 path/tag/session binding')
  require(actual['by_class']is by_class and actual['TPOT']['TPOT_denominator']==256 and actual['original_clock_summary']is ns['original_clock_summary'],'preserve category,charged256 and separate original publication clock result')
  require(actual['current_seven_realN_usage']is audit['seven_usage']and actual['actual_compatibility_classification']==dict(old_padding_behavior_equal=False,numerical_variant_only=True),'retain current producer schema and honest numerical variant classification')
  require(actual['independent_terminal_audit_pending']is True and actual['adopted']is actual['formal_Goodput']is actual['whole_workload_AR_qualified']is False,'recorder cannot adopt or grant AR/formal claims')
  complete_gate=next(copy.deepcopy(n)for n in tree.body if isinstance(n,ast.Assert)and "r['status'] == 'complete'"in ast.unparse(n))
  for status in ('failed','running'):
   try:exec(compile(ast.Module(body=[complete_gate],type_ignores=[]),str(SCRIPT)+'::actual-terminal-status-gate','exec'),dict(r=dict(status=status,errors=[],child=dict(exit_code=0))))
   except AssertionError:pass
   else:raise RuntimeError('noncomplete terminal cannot produce GP record')
  for p,h in constants.items():require(sha(p)==h,'source pin still exact: '+p)
  result.update(status='passed',whole_parent_projection=True,original_function_AST_equal=True,
   real_CLI_help=dict(exit_code=help_result.returncode,stdout=help_result.stdout),missing_authorization_rejected=True,
   wrong_session_rejected_before_artifact_access=True,actual_result_constructor_schema_executed=True,
   incomplete_and_failed_do_not_assign_GP=True,synthetic_schema_carrier_is_not_model_or_trace_evidence=True,
   source_pins_count=len(constants),current_trace_or_receipt_accessed=False)
 except BaseException:result['error']=traceback.format_exc()
 result.update(source_after={str(p):sha(p)for p in paths},elapsed_seconds=time.perf_counter()-start)
 require(result['source_before']==result['source_after'],'current source unchanged in same run')
 target=folder/'RESULT.json';target.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps(dict(path=str(target),sha256=sha(target),status=result['status'],elapsed_seconds=result['elapsed_seconds'])))
 if result['status']!='passed':print(result['error']);return 1
 return 0
if __name__=='__main__':raise SystemExit(run())
