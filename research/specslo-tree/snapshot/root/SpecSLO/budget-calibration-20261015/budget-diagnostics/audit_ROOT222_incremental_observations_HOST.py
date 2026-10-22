"""Recheck real222 controlled history outputs and logits, never run a model."""
from pathlib import Path
import json,hashlib,datetime,argparse
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
def need(x,m):
 if not x:raise AssertionError(m)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--actual',required=True);parser.add_argument('--output',required=True);args=parser.parse_args();p=Path(args.actual)
 paths=[p/f'rank{r}-numeric.json'for r in range(4)];reports=[json.loads(q.read_text())for q in paths];fixture_path=B/'tree-incremental-common-KV-history-forensics-v222/FIXTURE222.json';fixture=json.loads(fixture_path.read_text());cases=[]
 for idx,case in enumerate(fixture['cases']):
  r=[v['cases'][idx]for v in reports];need(all(v['request_id']==case['request_id']for v in r),'all4 real input identities');leader=r[1]['detail'];routes=[]
  for j,mode in enumerate(('candidate','reference')):
   values=[v['detail']['routes'][j]for v in r[1:]];expected=case['actual_'+mode+'_token'];dots={}
   for row in values:
    need(row['mode']==mode and row['original_first_roots_equal']and row['future_canary_query_hidden_and_common_KV_exact'],'actual role, initial roots and future invariance')
    need(row['query_position']==len(case['common_context'])-1 and row['query_token']==case['common_context'][-1],'same common last input')
    need(row['token']==values[0]['token']and row['BF16_top8']==values[0]['BF16_top8'],'all3 global target heads agree')
    for token,value in row['diagnostic_FP32_scalar_dots_local_shard'].items():
     need(token not in dots,'disjoint vocabulary shards');dots[token]=value
   ranking=sorted(dots,key=lambda t:(-dots[t],int(t)));scores=values[0]['BF16_top8']['values']
   routes.append(dict(route=mode,query_position=values[0]['query_position'],query_token=values[0]['query_token'],actual_neural_token=values[0]['token'],
    original_recorded_token=expected,reproduces_original_recorded_token=values[0]['token']==expected,original_BF16_top2_gap=scores[0]-scores[1],
    diagnostic_scalar_FP32_top2=[dict(token=int(t),value=dots[t])for t in ranking[:2]],
    scalar_dots_are_partial_vocab_only=True,original_first_roots_equal=True,exact_future_canary=True,frames=values[0]['frames']))
  cases.append(dict(request_id=case['request_id'],routes=routes,original_tokens_differ=case['actual_candidate_token']!=case['actual_reference_token'],
   actual_controlled_history_tokens_differ=routes[0]['actual_neural_token']!=routes[1]['actual_neural_token'],
   hidden_equal=leader['hidden_equal'],hidden_max_abs=leader['hidden_max_abs'],query_KV_different_indices=leader['query_KV_different_indices'],
   query_first_different_layer_zero_based=leader['query_first_different_layer'],complete_common_prefix_KV_different_indices=leader['complete_common_prefix_KV_different_indices'],
   common_prefix_first_different_layer_zero_based=leader['prefix_first_different_layer']))
 result=dict(schema='ROOT222.independent-controlled-history-observation-audit.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='observations_audited',
  source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest()for q in[Path(__file__),fixture_path,*paths]},cases=cases,
  original_differences_reproduced=sum(all(r['reproduces_original_recorded_token']for r in c['routes'])for c in cases),
  limits=['Only2 specific first-difference contexts; not proof for all177 requests.','Both paths teacher-force previously recorded histories; not independent AR.',
  'Candidate unaccepted future input tokens were not retained in original run; current diagnostic substitutes known completion tokens and checks future canary.',
  'No draft, verifier or scheduler runs during replay, but initial prompt prefill uses the original all4 helper.',
  'FP32 scalar dots use selected token IDs and the same BF16 hidden/weights; no model or sampler precision change.'],
  NPU_used_by_this_auditor=False,performance=False,independent_AR_qualified=False,formal_Goodput=False,adopted=False)
 out=Path(args.output);need(not out.exists(),'fresh retained HOST arithmetic');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],original_differences_reproduced=result['original_differences_reproduced'],cases=cases)))
if __name__=='__main__':main()
