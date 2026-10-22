"""Independent readonly arithmetic for four FULL9 contexts and dynamic body cost."""
from pathlib import Path
import json,hashlib,datetime,argparse,statistics

def need(x,m):
 if not x:raise AssertionError(m)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--actual',required=True);parser.add_argument('--output',required=True);args=parser.parse_args();P=Path(args.actual);paths=[P/f'rank{r}-numeric.json'for r in range(4)];reports=[json.loads(p.read_text())for p in paths];cases=[];costs=[]
 need([v['rank']for v in reports]==list(range(4)),'actual4 roles')
 for idx,case in enumerate(reports[1]['cases']):
  fresh=case['kind']=='fresh';rows=case['detail']['frames'if fresh else'routes'];summary=[]
  for j,row in enumerate(rows):
   keys=('graph_token','diagnostic_refined_token','diagnostic_tie_count')if fresh else('token','diagnostic_refined_token','diagnostic_tie_count')
   for report in reports[2:]:
    other=report['cases'][idx]['detail']['frames'if fresh else'routes'][j]
    need(all(row[k]==other[k]for k in keys),'all3 targets agree global heads')
   summary.append(dict(variant=row['name']if fresh else row['mode'],original_token=row['graph_token']if fresh else row['token'],refined_token=row['diagnostic_refined_token'],BF16_top_ties=row['diagnostic_tie_count']))
  if fresh:
   pairs=case['detail']['pairs'];equality=dict(query_hidden_all_variants_equal=all(p['graph_hidden_equal']for p in pairs),query_KV_all_variants_equal=all(p['graph_query_KV_equal']for p in pairs),pairs=pairs)
  else:
   detail=case['detail'];equality={k:detail[k]for k in('hidden_equal','hidden_max_abs','query_KV_different_indices','query_first_different_layer','complete_common_prefix_KV_different_indices','prefix_first_different_layer')}
   need(all(row['current_real16_bucketed_eager_graph_exact']and row['future_canary_query_hidden_and_common_KV_exact']for row in rows),'real16 hidden/KV eager reference and future canaries')
  cases.append(dict(case=case['case'],kind=case['kind'],request_id=case['request_id'],variants=summary,
   refined_tokens_same_across_variants=len({r['refined_token']for r in summary})==1,numerical_equality=equality))
 for rank,report in enumerate(reports[1:],start=1):
  histories=[c for c in report['cases']if c['kind']=='history'];c=histories[-1]['detail']['routes'][-1]['dynamic_attention_body_cost'];samples=c['samples'];need(len(samples)==12,'3ABBA blocks')
  for b in range(3):
   for j,mode in enumerate(('causal','FULL9','FULL9','causal')):
    x=samples[4*b+j];need((x['block'],x['cell'],x['mode'],x['replays'])==(b,j,mode,6),'predeclared interleaved dynamic body samples')
    need(x['execution']['mode']=='replay'and x['device_ms_per_call']>0 and x['host_wall_ms_per_call']>0,'actual positive original replay timing')
    need(x['counter_delta']['updates']+x['counter_delta']['skips']==6,'each original replay updates or skips its attention tasks')
  med={mode:{clock:statistics.median(x[clock]for x in samples if x['mode']==mode)for clock in('host_wall_ms_per_call','device_ms_per_call')}for mode in('causal','FULL9')}
  need(c['medians']==med,'exact median arithmetic')
  costs.append(dict(rank=rank,medians=med,samples=samples,updates_by_mode={mode:sum(x['counter_delta']['updates']for x in samples if x['mode']==mode)for mode in med},skips_by_mode={mode:sum(x['counter_delta']['skips']for x in samples if x['mode']==mode)for mode in med}))
 out=Path(args.output);need(not out.exists(),'fresh independent output');result=dict(schema='ROOT225.independent-FULL9-numeric-and-dynamic-body-cost.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='bounded_observations_arithmetic_audited',
  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in[Path(__file__),*paths]},cases=cases,costs=costs,
  limitations=['Four selected contexts only, including fresh common prefix at new224 difference148; not a full independentAR proof.',
   'Two incremental histories teacher-force recorded common outputs; fresh prefill does not reproduce original incremental KV.',
   'Original sampler stays installed; refined tokens are a separate diagnostic observer.',
   'Body timings include original metadata copy/replay/task updates with prebuilt metadata, exclude LM-head and all serving/scheduling work; not whole Goodput.',
   'HOST checks raw arithmetic and source-bound device counters; no neural or physical time reexecution.'],NPU_used_by_this_auditor=False,formal_Goodput=False,adopted=False)
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],cases=[{k:v for k,v in c.items()if k!='numerical_equality'}for c in cases],costs=[{k:v for k,v in c.items()if k!='samples'}for c in costs])))
if __name__=='__main__':main()
