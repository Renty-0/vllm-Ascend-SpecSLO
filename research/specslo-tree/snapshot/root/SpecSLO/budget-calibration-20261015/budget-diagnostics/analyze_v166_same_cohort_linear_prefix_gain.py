"""HOST-only same-executed-layout prefix cuts, never a new model/latency result."""
from pathlib import Path
from collections import Counter,defaultdict
import datetime,gzip,hashlib,json,statistics
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-active-window-full-v166';tag='active32-24-16-v166-full463-v1'
source=P/'actual'/tag/'windows/window32/full463/rank1-full463-actual-trace.json.gz'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(source)=='5760a56cd565666a873486ec94523a3667508d7ff2bf304883a154dc6c78fb83'
a=json.loads(gzip.decompress(source.read_bytes()));results=a['generate_results'];calls=a['normal_target_calls']
first=[e for e in a['stream_events'] if 'first_token_wall_seconds160' in e]
events=[e for e in a['stream_events'] if 'first_token_wall_seconds160' not in e]
assert len(results)==len(first)==463 and all(len(e['token_ids'])==1 for e in first)
assert sum(len(e['token_ids']) for e in a['stream_events'])==463*256
flat=[i for c in calls for i in c['sequence_ids'][:c['N']]]
assert [e['request_index'] for e in events]==flat
by=defaultdict(list);byctx=defaultdict(list);byowner=defaultdict(list);skip=Counter();pos=0
for c in calls:
 for owner,budget,context in zip(c['sequence_ids'][:c['N']],c['budgets'],c['contexts'],strict=True):
  event=events[pos];pos+=1
  assert event['request_index']==owner
  if budget!=4:skip['not_B4']+=1;continue
  if c['family']!='causal':skip['not_all_contiguous_causal']+=1;continue
  if event['finished']:skip['terminal_right_censoring']+=1;continue
  delivered=len(event['token_ids']);assert 1<=delivered<=5
  accepted=delivered-1
  cuts=[min(accepted,n)+1 for n in range(1,5)]
  assert cuts[-1]==delivered
  cls=results[owner]['slo_class'];by[cls].append((owner,accepted,cuts));byowner[owner].append(cuts)
  key='0..255' if context<256 else '256..511' if context<512 else '512..1023' if context<1024 else '1024+'
  byctx[(cls,key)].append((owner,accepted,cuts))
assert pos==len(events)
def describe(rows):
 return dict(context_rows=len(rows),distinct_requests=len({r[0] for r in rows}),
  accepted_primary_nodes_histogram=dict(sorted(Counter(r[1] for r in rows).items())),
  expected_effective_tokens_at_n={str(n):statistics.mean(r[2][n-1] for r in rows) for n in range(1,5)},
  exact_lower_prefix_delivery_formula='g_n(context)=1+min(actual accepted primary nodes,n), n=1..4')
classes={c:describe(by[c]) for c in ('tight','normal','loose')}
request_equal={}
for cls in ('tight','normal','loose'):
 owners=sorted({r[0] for r in by[cls]})
 request_equal[cls]={str(n):statistics.mean(statistics.mean(r[n-1] for r in byowner[i]) for i in owners) for n in range(1,5)}
v=dict(schema='ROOT.V166.same-cohort-executed-layout-linear-prefix-gain.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 source=dict(path=str(source),sha256=sha(source)),script=dict(path=str(Path(__file__)),sha256=sha(__file__)),
 status='completed_readonly_HOST_analysis',actual_target_calls=len(calls),actual_normal_owner_rounds=len(events),
 exact_target_owner_and_stream_order_equal=True,actual_initial_single_token_callbacks=463,
 admitted_B4_causal_nonterminal_rows=sum(len(v) for v in by.values()),excluded=dict(skip),
 by_class_round_weighted=classes,by_class_equal_request_weighted=request_equal,
 by_class_and_context_length={f'{c}/{k}':describe(v) for (c,k),v in sorted(byctx.items())},
 interpretation='Paired mathematical truncations on the same executed B4 contexts and physical layout; n4 is observed delivery. Lower-n cuts do not execute the alternate mapper or new autoregressive trajectory.',
 limitations=['Actual job166 failed after retaining full32; full trace itself independently audited.',
 'No candidate/future target tokens generated or changed; no device query, replay, or model call.',
 'Only linear prefix budgets1..4; B5 is a real branch and cannot be extrapolated from this formula.',
 'Actual selector/state/context distribution; repeated contexts within each request are correlated; no iid confidence interval.',
 'Different physical M or altered future history can change numerical heads and context distribution; these are not measured counterfactual runs.',
 'A per-class g curve does not price target family/M/S, graph padding, draft steps, prefill interference or SLO queueing.'],
 policy_changed=False,model_or_NPU_used=False,new_performance_measurement=False,formal_Goodput=False,adopted=False)
p=D/'ROOT_V166_SAME_COHORT_LINEAR_PREFIX_GAIN.json';assert not p.exists();p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(path=str(p),sha256=sha(p),admitted=v['admitted_B4_causal_nonterminal_rows'],by_class=classes),ensure_ascii=False))
