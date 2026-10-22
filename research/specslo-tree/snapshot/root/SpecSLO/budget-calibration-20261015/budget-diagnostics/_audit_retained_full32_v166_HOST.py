import ast,gzip,hashlib,importlib.abc,importlib.util,json,math,statistics,sys,tarfile
from pathlib import Path
from datetime import datetime,timezone
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-active-window-full-v166';BASE=B/'tree-high-normal-cap4-native-families-v148';tag='active32-24-16-v166-full463-v1';A=P/'actual'/tag
blocked={'torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'}
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in blocked:raise ImportError('HOST backend forbidden: '+fullname)
sys.meta_path.insert(0,Firewall())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(x,m):
 if not x:raise AssertionError(m)
def read(p):return json.loads(Path(p).read_bytes())
def desc(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def mod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def load(p):
 raw=p.read_bytes();data=gzip.decompress(raw);need(len(data)<=256*1024**2,'expandedbound');return json.loads(data),dict(**desc(p),uncompressed_bytes=len(data))
rp=P/f'receipt-{tag}.json';r=read(rp);fp=P/'FROZEN_SOURCE166.json';f=read(fp)
need(r['status']=='failed' and r['tag']==tag and r['action']=='active-window-full463','actual failed epoch')
need(r['readiness']['freeze_sha256']==sha(fp) and r['readiness']['CPU_sha256']==sha(P/'CPU_RESULT166.json'),'receipt freeze/CPU')
pins={}
for field in ('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):
 for p,h in f[field].items():need(p not in pins or pins[p]==h,'compatible overlaps');pins[p]=h
for p,h in pins.items():need(Path(p).is_file() and sha(p)==h,'current source '+p)
need(sha(f['root_review']['path'])==f['root_review']['sha256'],'rootreview pin')
archive=r['readiness']['archive'];need(sha(archive['path'])==archive['sha256'],'archive digest');members=0
with tarfile.open(archive['path'],'r:gz') as tf:
 for member in tf:
  need(member.isfile(),'regular archive');data=tf.extractfile(member).read();members+=1;p=Path('/'+member.name.lstrip('/'));need(p.is_file() and p.read_bytes()==data,'archive member '+member.name)
need(members==archive['members']==77,'77 members')
for label in ('pre_verifier','post_verifier'):
 v=r[label];need(v['exit_code']==0 and sha(v['path'])==v['sha256'],'baseline log');q=read(v['path']);need(q==v['actual_result'] and q['status']=='pass' and not q['errors'] and (q['checked_frozen_files'],q['checked_external_files'],q['checked_packages'])==(10496,15,212),'baseline prepost')
need(r['sources_unchanged'] is True and r['owned_remaining']==[] and r['child']['exit_code']==1,'receipt terminal')
for label in ('devices_before','devices_after'):
 v=r[label];need(v['all_idle'] is v['four_910B2'] is True and set(v['processes'])=={'2','5','6','7'},'saved idle')
 for x in v['processes'].values():need(x['exit_code']==0 and x['idle'] is True and x['output'].strip()=='No process in device.','saved processes')
wp=A/'worker-exit.json';need(read(wp)==dict(closed=True,all_exited=True),'worker closed');errors=[]
for rank in range(4):
 p=A/f'rank{rank}-last-error.json';v=read(p);need(v['rank']==rank and "NameError: name 'sha' is not defined" in v['error'] and 'V166/V160/window32/full463/unchanged-clock-metrics-and-actual-TTFT' in v['error'],'common error');errors.append(desc(p))
ap=BASE/'audit_production_v101.py';original=mod('v166_independent_original_audit',ap);mp=BASE/'no_mixed_prefill_v143.py';mode=mod('v166_independent_mode_audit',mp)
manifest='/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/manifest.jsonl';rows,prefix=original.manifest_rows(manifest,463);traces=[];artifacts=[]
for rank in range(4):
 t,d=load(A/'windows/window32/full463'/f'rank{rank}-full463-actual-trace.json.gz');traces.append(t);artifacts.append(d)
full=original.audit_trace_set(traces,rows,prefix,zero=True);standalone=[mode.audit_production_mode_trace(t,463) for t in traces];need(all(x==standalone[0] for x in standalone),'standalone all4')
for t in traces:
 need(t['admission_window160']==32 and t['configured_capacity']==64 and t['active_peak']==32 and t['actual_window_covered160'] is True and t['actual_capacity64_scope_completed'] is False,'honest32 scope')
 need(t['sealed_before'] is t['sealed_after'] is True and t['cache_released'] is True,'actual seal/cachefree')
 need(all(b<=(4 if c['N']>8 else 5) for c in t['normal_target_calls'] for b in c['budgets']),'actual >8cap4')
cp=[];cpdesc=[]
for rank in range(4):
 p=A/f'rank{rank}-window-checkpoint.json.gz';v,d=load(p);cp.append(v);cpdesc.append(d)
schedule=[(w,'warmup65') for w in (32,24,16)]+[(w,role) for role in ('zero65-1','zero65-2') for w in (32,24,16)]
need(all([(t['window'],t['role']) for t in c['traces']]==schedule and c['metrics']==[] for c in cp),'nine checkpoints metrics absent')
warm=[];zeros=84
for idx,(window,role) in enumerate(schedule):
 raw=[]
 for rank in range(4):
  env=cp[rank]['traces'][idx];t=original.load_artifact(env['original_trace']['artifact'],A);need(t['admission_window160']==window and t['sealed_before'] is t['sealed_after'] is False,'preseal65');raw.append(t)
 rows65,prefix65=original.manifest_rows(manifest,65);value=original.audit_trace_set(raw,rows65,prefix65,zero=role!='warmup65')
 for t in raw:mode.audit_production_mode_trace(t,65)
 if role!='warmup65':zeros+=84
 warm.append(dict(window=window,role=role,requests=value['output']['requests'],output_tokens=value['output']['output_tokens'],original_all4_audit_passed=True,Native21_zero_all4=role!='warmup65'))
need(zeros==588,'all21zero588')
need(all(not any((A/'windows'/f'window{w}'/'full463').glob('*actual-trace*')) for w in (24,16)),'later full absent')
t=traces[1];results=t['generate_results'];first={};seen=set()
for e in t['stream_events']:
 owner=e['request_index']
 if owner not in seen:need('first_token_wall_seconds160' in e,'first callback');first[owner]=e['first_token_wall_seconds160'];seen.add(owner)
 else:need('first_token_wall_seconds160' not in e,'one first callback')
need(set(first)==set(range(463)),'463 callbacks')
for i,row in enumerate(results):need(row['arrival_ts']==t['absolute_arrivals'][i],'unchanged absolute arrivals')
ttft=[(first[i]-row['arrival_ts'])*1000 for i,row in enumerate(results)];need(all(math.isfinite(v) and v>=0 for v in ttft),'finite TTFT')
def pct(v):
 a=sorted(v);p=(len(a)-1)*.9;lo=math.floor(p);hi=math.ceil(p);return dict(median_ms=statistics.median(a),P90_ms=a[lo]+(a[hi]-a[lo])*(p-lo))
clockp=D/'analyze_completed_tree_full463_cost.py';tree=ast.parse(clockp.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('need','clocks')];need(len(nodes)==2,'two unchanged original clock funcs');ns=dict(math=math,statistics=statistics,CLASSES=('tight','normal','loose'));exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(clockp)+'::independent','exec'),ns);metrics=ns['clocks'](t)
metrics['TTFT']=dict(samples=463,**pct(ttft),by_class={c:pct([ttft[i] for i,row in enumerate(results) if row['slo_class']==c]) for c in ('tight','normal','loose')},clock='first original callback time.time minus unchanged absolute arrival_ts')
need(metrics['Native_paper']['attained']==full['output']['attained']==191 and metrics['raw_tokens_per_second']==full['output']['raw_tokens_per_second'] and metrics['Native_paper']['Goodput']==full['output']['goodput_tokens_per_second'],'independent two arithmetic equality')
rootp=D/'ROOT_V166_RETAINED_FULL32_ANALYSIS.json';root=read(rootp);rootm=root['completed_full_windows'][0]['metrics'];need(rootm['Native_paper']==metrics['Native_paper'] and rootm['raw_tokens_per_second']==metrics['raw_tokens_per_second'] and rootm['TTFT160']['median_ms']==metrics['TTFT']['median_ms'] and rootm['TTFT160']['P90_ms']==metrics['TTFT']['P90_ms'],'ROOT match after independent recompute')
need(not any(x.split('.')[0] in blocked for x in sys.modules),'no backend imports')
output=dict(schema='V166.independent-retained-full32-terminal-audit.v1',status='passed_retained_full32_audit',candidate=str(P),tag=tag,created_UTC=datetime.now(timezone.utc).isoformat(),read_only_HOST=True,NPU_or_model_import_or_query=False,tool=desc(Path(__file__)),
 whole_job=dict(status='failed',child_exit_code=1,first_actual_failure_stage='V166/V160/window32/full463/unchanged-clock-metrics-and-actual-TTFT',all4_error="NameError: name 'sha' is not defined",full_windows_completed=[32],full_windows_not_run=[24,16],complete_three_window_job_passed=False),
 original_audit=dict(source=desc(ap),function='audit_trace_set',unmodified_function=True,all4_full463_passed=True,manifest_prefix_sha256=prefix,Native21_full_zero_subtractions=84,Native21_six_zero65_subtractions=504,Native21_total_zero_subtractions=zeros,actual_transactions=full['transactions'],normal_target_calls=full['normal_target_calls'],actual_peak_active=full['actual_peak_active'],preseal_traces=warm),
 artifacts=artifacts,checkpoints=cpdesc,metrics=metrics,output=dict(requests=463,tokens=full['output']['output_tokens'],attained=191,by_class=full['output']['by_class'],token_output_sha256=full['output']['output_sha256'],stream_client_all256=True),
 provenance=dict(receipt=desc(rp),freeze=desc(fp),CPU=desc(P/'CPU_RESULT166.json'),selected_current_SHA_verified=len(pins),archive=dict(**archive,actual_regular_members_verified=members),source_prepost_receipt_unchanged=True,current_selected_bytes_equal_freeze=True,baseline_prepost=dict(frozen_files=10496,external_files=15,packages=212,status='pass'),worker_exit=desc(wp),last_errors=errors,saved_prepost_four_devices_idle=True,owned_remaining=[],new_hardware_query_performed=False),
 independence=dict(original_trace_stream_arithmetic_recomputed=True,actual_first_callback_TTFT_recomputed=True,firstwall_not_substituted_for_Native_TPOT=True,current_source_and_saved_prepost_only=True,baseline_verifier_not_rerun=True,root_summary=desc(rootp),root_summary_matches_independent_recompute=True,hidden_KV_or_AR_recomputed=False,own_B4_component_numerics_not_replayed_in_this_audit=True),
 limitations=dict(formal_Goodput=False,adopted=False,whole_workload_AR_qualified=False,complete_P166_job_passed=False,Native_finish_start_formula_reconstructed=False,scope='Retained full32 trace/output/counter/protocol/clock and terminal-source/cleanup audit; no numerical qualification or result for unrun windows'))
dest=D/'V166_INDEPENDENT_RETAINED_FULL32_TERMINAL_AUDIT.json';need(not dest.exists(),'no overwrite');dest.write_text(json.dumps(output,ensure_ascii=False,sort_keys=True,indent=2)+'\n');print(json.dumps(dict(path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,raw=metrics['raw_tokens_per_second'],Goodput=metrics['Native_paper']['Goodput'],attained=191,whole_job='failed',source_pins=len(pins),zero_subtractions=zeros)))
