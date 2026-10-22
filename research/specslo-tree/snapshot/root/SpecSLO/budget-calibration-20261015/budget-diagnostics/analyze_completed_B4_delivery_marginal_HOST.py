"""Old-trajectory B4 delivery arithmetic, never new-B3 performance prediction."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import gzip,json,hashlib
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p);return json.loads(gzip.decompress(p.read_bytes())if p.suffix=='.gz'else p.read_bytes())
def one(epoch,directory,tag,audit_sha):
 audit_path=D/('V'+str(epoch)+'_INDEPENDENT_TERMINAL_FULL463_AUDIT.json')
 assert sha(audit_path)==audit_sha
 audit=read(audit_path);assert audit['status']=='passed_terminal_component_and_full463_audit'and not audit['errors']
 manifest=audit['raw_manifest'];assert sha(manifest['path'])==manifest['sha256']
 raw=B/directory/'actual'/tag/'windows/window32/full463/rank1-full463-actual-trace.json.gz'
 assert read(manifest['path'])[str(raw)]['sha256']==sha(raw)
 x=read(raw);indices={e['request_id']:e['request_index']for e in x['stream_events']}
 assert len(indices)==463 and set(indices.values())==set(range(463))
 classes={indices[r['request_id']]:r['slo_class']for r in x['generate_results']}
 calls=defaultdict(list);events=defaultdict(list);first=set()
 for call in x['normal_target_calls']:
  for j in range(call['N']):calls[call['sequence_ids'][j]].append((call['budgets'][j],call['contexts'][j]))
 for event in x['stream_events']:
  i=event['request_index']
  if i in first:events[i].append((len(event['token_ids']),event['finished']))
  else:first.add(i);assert len(event['token_ids'])==1
 assert set(calls)==set(events)==first==set(range(463))
 summaries={};all_budgets=Counter();b4_all=Counter()
 for label in ('tight','normal','loose'):
  histogram=Counter();censored=Counter();owner_rounds=0
  for i in range(463):
   assert len(calls[i])==len(events[i])
   for j,((budget,context),(delivered,finished))in enumerate(zip(calls[i],events[i])):
    assert 1<=delivered<=budget+1
    if j+1<len(calls[i]):assert calls[i][j+1][1]==context+delivered
    if classes[i]!=label:continue
    owner_rounds+=1;all_budgets[budget]+=1
    if budget==4:
     b4_all[delivered]+=1
     (censored if finished else histogram)[delivered]+=1
   assert 1+sum(n for n,_ in events[i])==256
  n=sum(histogram.values());assert n>0
  summaries[label]=dict(owner_rounds=owner_rounds,nonterminal_B4_rounds=n,
   nonterminal_B4_delivered_histogram=dict(sorted(histogram.items())),terminal_censored_B4_histogram=dict(sorted(censored.items())),
   actual_B4_mean_delivered_on_its_old_nonterminal_histories=sum(k*v for k,v in histogram.items())/n,
   truncate_at4_old_heads_arithmetic=sum(min(k,4)*v for k,v in histogram.items())/n,
   actual_B4_fifth_output_frequency=histogram[5]/n,
   interpretation='truncate_at4 is only min(actual old B4 publication,4); not an observed B3 model, newM64, future history or a fitted g3 curve')
 return dict(epoch=epoch,raw=dict(path=str(raw),sha256=sha(raw)),audit=dict(path=str(audit_path),sha256=audit_sha),
  all_owner_rounds=sum(map(len,calls.values())),all_budget_histogram=dict(sorted(all_budgets.items())),
  all_B4_publication_histogram=dict(sorted(b4_all.items())),classes=summaries,
  joined_by_actual_owner_and_monotone_context=True,all463_output256=True)
result=dict(schema='ROOT.completed-B4-delivery-marginal.HOST.v1',UTC=datetime.now(timezone.utc).isoformat(),status='completed_old_trajectory_arithmetic_only',
 cases=[one(197,'tree-empty-active-cohort-causal80-active32-full-v197','empty-cohort-v197-full463-v1',
 'b6e1b3465c1c9b0d3353b9b3b60876aedd318cc75b00e5a3ab2849ce5ec62a99'),
 one(203,'tree-empty-active-slo-admission1-causal80-active32-full-v203','empty-cohort-slo1-v203-full463-v1',
 'ea3deedbef72988c3aa4b18fe9b13b627d61a1be0d2cd878b194f4e0fbb4e29d')],
 limitations=['No new model or B3 performance was measured. Changed N/M shapes, draft length, padding, scheduler/E behavior and numerical trajectories prevent treating min(old delivery,4) as actual new g3 or predicting GP.',
 'Terminal callbacks are separately censored; nonterminal conditional means describe old B4-selected histories, not random class-wide samples.',
 'The joined published event length includes original clipping and output protocol. Actual target/draft logits and hidden/KV are not reconstructed here.'],NPU_used=False,Torch_imported=False,models_run=False,new_g_installed=False,formal_Goodput=False)
out=D/'COMPLETED_197_203_B4_DELIVERY_MARGINAL_HOST.json'
with out.open('x')as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
print(json.dumps(dict(output=str(out),sha256=sha(out),cases=[dict(epoch=c['epoch'],classes=c['classes'])for c in result['cases']]),ensure_ascii=False))
