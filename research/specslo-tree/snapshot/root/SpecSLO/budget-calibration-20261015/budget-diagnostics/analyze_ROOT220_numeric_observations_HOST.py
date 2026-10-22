"""Independent integer/logit arithmetic from bounded real220 observations."""
from pathlib import Path
import json,hashlib,datetime,argparse
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
def need(x,m):
 if not x:raise AssertionError(m)
def main():
 p=argparse.ArgumentParser();p.add_argument('--actual',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.actual)
 paths=[root/f'rank{r}-numeric.json'for r in range(4)];reports=[json.loads(p.read_text())for p in paths]
 need([r['rank']for r in reports]==list(range(4)),'all4 actual roles')
 cases=[]
 for i in range(3):
  source=[r['cases'][i]for r in reports];leader=source[1];frames=leader['detail']['frames'];rows=[]
  for j,frame in enumerate(frames):
   values=[]
   for mode in('eager','graph'):
    dots={}
    for role in range(1,4):
     other=source[role]['detail']['frames'][j]
     for field in('name','shape','query_row','query_position','query_input_token','eager_token','graph_token','eager_top8','graph_top8'):
      need(other[field]==frame[field],'same global result across actual target ranks '+field)
     local={int(k):v for k,v in other['diagnostic_FP32_scalar_dots_local_shard'][mode].items()}
     need(not(set(local)&set(dots)),'disjoint actual vocabulary shards');dots.update(local)
    ranking=sorted(dots,key=lambda t:(-dots[t],t));top=frame[mode+'_top8'];gap=top['BF16_values'][0]-top['BF16_values'][1]
    need(gap>=0,'decreasing saved original top8 scores')
    values.append(dict(mode=mode,original_BF16_greedy_token=frame[mode+'_token'],original_BF16_top2_gap=gap,
      diagnostic_scalar_FP32_top2=[dict(token=t,value=dots[t])for t in ranking[:2]],
      scalar_FP32_dot_margin=dots[ranking[0]]-dots[ranking[1]],
      scalar_dots_are_partial_vocab_diagnostic_not_replacement_sampler=True))
   rows.append(dict(name=frame['name'],query_position=frame['query_position'],query_input_token=frame['query_input_token'],
    original_actual_tokens=dict(candidate=frame['expected_actual_candidate'],reference=frame['expected_actual_reference']),
    eager_graph_hidden_equal=frame['eager_graph_hidden_bitwise_equal'],eager_graph_hidden_max_abs=frame['eager_graph_hidden_max_abs'],
    eager_graph_query_KV_equal=frame['eager_graph_query_KV_equal'],graph_execution=frame['graph_execution'],heads=values))
  need(all(s['case']==leader['case']and s['request_id']==leader['request_id']for s in source),'same3 real diagnostic requests')
  repeat=leader['detail']['pairs'][-1] if len(frames)==4 else next(p for p in leader['detail']['pairs']if p['a']==frames[0]['name']and p['b']==frames[-1]['name'])
  # QUERY1 is first frame and QUERY1_REPEAT is last, even with4 total frames.
  repeat=next(p for p in leader['detail']['pairs']if p['a']==frames[0]['name']and p['b']==frames[-1]['name'])
  need(repeat['same_query_token_and_position'],'repeat same input and position')
  cases.append(dict(case=leader['case'],request_id=leader['request_id'],frames=rows,pairs=leader['detail']['pairs'],
    same_variant_repeat_hidden_exact=dict(eager=repeat['eager_hidden_equal'],graph=repeat['graph_hidden_equal']),
    same_variant_repeat_query_KV_exact=dict(eager=repeat['eager_query_KV_equal'],graph=repeat['graph_query_KV_equal']),
    original_affected_prefix_unchanged=True))
 out=Path(a.output);need(not out.exists(),'fresh independent observation audit')
 report=dict(schema='ROOT220.independent-numeric-observation-arithmetic.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in[Path(__file__),*paths]},status='observations_audited',cases=cases,
  limitations=['Only3 selected first differences are probed; not an explanation of all177 differences.',
   'Fresh teacher-forced common prefix prefill differs from incremental production/reference history; actual recorded tokens need not reproduce.',
   'Scalar FP32 dots use the same BF16 hidden states and weights and selected token IDs; they are diagnostic, not a new full FP32 model or sampler.',
   'No independent full463 AR or performance qualification is provided.'],NPU_used_by_this_auditor=False,formal_Goodput=False,adopted=False)
 out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=report['status'],output=str(out),cases=3)))
if __name__=='__main__':main()
