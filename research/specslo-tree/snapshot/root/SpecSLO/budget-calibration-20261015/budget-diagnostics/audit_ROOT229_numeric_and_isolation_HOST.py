"""Independent source-bound arithmetic audit; never executes the model."""
import argparse,datetime,hashlib,json,statistics
from pathlib import Path
def need(v,m):
    if not v:raise AssertionError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--actual',required=True);parser.add_argument('--receipt',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    P=Path(args.actual);receipt_path=Path(args.receipt);receipt=json.loads(receipt_path.read_text())
    need(receipt['status']=='complete'and receipt['sources_unchanged']and not receipt['owned_remaining']and receipt['devices_after']['all_idle'],'original supervisor completed and restored all four idle devices')
    need(receipt['pre_verifier']['actual_result']['status']==receipt['post_verifier']['actual_result']['status']=='pass','actual pre/post baseline checks')
    paths=[P/f'rank{r}-numeric.json'for r in range(4)];reports=[json.loads(p.read_text())for p in paths]
    need([r['rank']for r in reports]==list(range(4))and all(r['original_greedy_sampler_unchanged']and not r['observer_installed']for r in reports),'original sampler on all four actual roles')
    cases=[]
    for i,c in enumerate(reports[1]['cases']):
        fresh=c['kind']=='fresh';rows=c['detail']['frames'if fresh else'routes'];tokens=[]
        for j,row in enumerate(rows):
            token_key='graph_token'if fresh else'token'
            for other in reports[2:]:
                x=other['cases'][i]['detail']['frames'if fresh else'routes'][j]
                need(all(row[k]==x[k]for k in(token_key,'diagnostic_refined_token','diagnostic_tie_count')),'all three target ranks agree on global selection')
            tokens.append(dict(variant=row['name'if fresh else'mode'],original=row[token_key],refined=row['diagnostic_refined_token'],BF16_tie_count=row['diagnostic_tie_count']))
        if fresh:
            detail=dict(hidden_all_variants_equal=all(x['graph_hidden_equal']for x in c['detail']['pairs']),query_KV_all_variants_equal=all(x['graph_query_KV_equal']for x in c['detail']['pairs']),pairs=c['detail']['pairs'])
        else:
            for row in rows:
                m=row['same_metadata_eager_graph_observation'];need(m['original_matcher_passed']and m['identical_exact_metadata_used']and not m['bucket_helper_used']and row['future_canary_query_hidden_and_common_KV_exact'],'same exact metadata and future canaries')
            detail={k:c['detail'][k]for k in('hidden_equal','hidden_max_abs','query_KV_different_indices','complete_common_prefix_KV_different_indices')}
        cases.append(dict(case=c['case'],tokens=tokens,numerical_observation=detail))
    ablations=[]
    names=['q1_M40_B5','q1_M80_B5','q1_M80_B4','q5_M80_B5','q5_M80_B4','q1_M40_B5_REPEAT']
    for i,c in enumerate(reports[1]['ablations']):
        need([v['name']for v in c['variants']]==names,'predeclared eager-only controls')
        for j,v in enumerate(c['variants']):
            need(v['eager_only'],'no extra graph qualification claimed for ablation')
            for other in reports[2:]:
                x=other['ablations'][i]['variants'][j];need(x['token']==v['token']and x['query_hidden']==v['query_hidden'],'replicated target hidden and original heads agree; shard KV deliberately not compared')
        ablations.append(c)
    costs=[]
    for rank,report in enumerate(reports[1:],start=1):
        c=[x for x in report['cases']if x['kind']=='history'][-1]['detail']['routes'][-1]['dynamic_attention_body_cost'];samples=c['samples']
        need(len(samples)==6 and c['zero_new_body_graph_captures']and len(c['eager_graph_qualification'])==2,'existing one-family graph, two independently eager-matched contexts')
        for i,s in enumerate(samples):
            need((s['block'],s['cell'],s['mode'],s['replays'])==(i//2,i%2,'q1_FULL1',6),'predeclared absolute timing cells')
            need(s['execution']['mode']=='replay'and s['host_wall_ms_per_call']>0 and s['device_ms_per_call']>0,'positive actual graph timing')
            need(sum(s['counter_delta'].values())==6,'each replay refreshed or skipped original task updates')
        med={clock:statistics.median(s[clock]for s in samples)for clock in('host_wall_ms_per_call','device_ms_per_call')}
        need(med==c['medians'],'independent exact median arithmetic')
        costs.append(dict(rank=rank,medians=med,samples=samples,eager_graph_qualification=c['eager_graph_qualification']))
    out=Path(args.output);need(not out.exists(),'fresh independent output')
    value=dict(schema='ROOT229.independent-numeric-and-eager-isolation-audit.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='bounded_observations_audited',
        source_sha256={str(p):sha(p)for p in[Path(__file__),receipt_path,*paths]},cases=cases,eager_ablations=ablations,existing_graph_absolute_cost=costs,
        limitations=['Selected controlled common histories only; no whole463 independent AR proof.','M/Q ablations use eager execution and two fresh prefixes.','Absolute body cost excludes prefill, LM-head, scheduling, and metadata construction; no comparative speed or Goodput claim.'],
        NPU_used_by_this_auditor=False,formal_Goodput=False,adopted=False)
    with out.open('x')as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=value['status'],cases=cases,ablation_tokens=[[(v['name'],v['token'])for v in c['variants']]for c in ablations],costs=[dict(rank=c['rank'],medians=c['medians'])for c in costs])))
if __name__=='__main__':main()
