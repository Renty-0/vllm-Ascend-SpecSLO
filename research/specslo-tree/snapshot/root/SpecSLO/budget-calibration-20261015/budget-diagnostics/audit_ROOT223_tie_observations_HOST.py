"""Read-only independent medians and original/refined diagnostic choices."""
from pathlib import Path
import argparse,json,hashlib,datetime,statistics

def need(x,m):
 if not x:raise AssertionError(m)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--actual',required=True);parser.add_argument('--output',required=True);args=parser.parse_args();p=Path(args.actual);paths=[p/f'rank{r}-numeric.json'for r in range(4)];reports=[json.loads(q.read_text())for q in paths]
 need([r['rank']for r in reports]==list(range(4)),'four actual original roles');cases=[];costs=[]
 for rank,report in enumerate(reports):
  need(report['observer_installed']is False and report['model_body_or_original_sampler_changed']is False,'uninstalled observer scope')
  if not rank:continue
  groups=[report['cases'][0]['detail']['frames']]+[c['detail']['routes']for c in report['cases'][1:]]
  for idx,rows in enumerate(groups):
   summary=[]
   for row in rows:
    original=row['graph_token']if idx==0 else row['token']
    summary.append(dict(name=row['name']if idx==0 else row['mode'],original_token=original,refined_token=row['diagnostic_refined_token'],top_BF16_ties=row['diagnostic_tie_count'],overflow=row['diagnostic_tie_overflow']))
    need(row['diagnostic_refined_token']==(30,1817,7196)[idx]and not row['diagnostic_tie_overflow'],'predeclared diagnostic winner')
    if 'isolated_head_cost'in row:
     cost=row['isolated_head_cost'];samples=cost['replay_samples'];need(len(samples)==12,'three ABBA blocks')
     for block in range(3):
      for cell,mode in enumerate(('original','refined','refined','original')):
       item=samples[block*4+cell];need((item['block'],item['cell'],item['mode'],item['replays'])==(block,cell,mode,20)and item['device_ms_per_replay']>0,'actual ordered positive ABBA sample')
     med={mode:statistics.median(v['device_ms_per_replay']for v in samples if v['mode']==mode)for mode in('original','refined')}
     need(cost['median_device_ms']==med and cost['extra_median_ms']==med['refined']-med['original'],'exact median and extra arithmetic')
     costs.append(dict(rank=rank,M=cost['M'],median_device_ms=med,extra_median_ms=cost['extra_median_ms'],validated_eager_graph_repeat=cost['eager_graph_and_repeat_exact'],samples=samples))
   cases.append(dict(rank=rank,case=idx,rows=summary))
 for idx in range(3):
  found=[c['rows']for c in cases if c['case']==idx];need(found[0]==found[1]==found[2],'all3 target ranks same original and refined choices')
 need(len(costs)==6,'M40/M80 on all3 target ranks')
 result=dict(schema='ROOT223.independent-tie-observer-and-head-median-audit.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='observations_and_cost_arithmetic_audited',
  source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest()for q in[Path(__file__),*paths]},cases=cases,costs=costs,
  limitations=['Only3 selected first-difference contexts; no full463 independent AR or performance qualification.',
   'Original model and sampler remain installed. The new tie rule is an observer, not an adopted generation policy.',
   'Device timing is source-bound isolated head graph event timing; HOST recomputes sample order and medians, not physical NPU durations.',
   'The refinement handles complete winning ties up to4 per shard and returns original choice on overflow; it does not guarantee numerical invariance for all contexts.'],
  NPU_used_by_this_auditor=False,production_sampler_changed=False,full_Goodput=None,formal_Goodput=False,adopted=False)
 out=Path(args.output);need(not out.exists(),'fresh independent arithmetic');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],costs=[{k:v for k,v in row.items()if k!='samples'}for row in costs],cases=cases[:3])))
if __name__=='__main__':main()
