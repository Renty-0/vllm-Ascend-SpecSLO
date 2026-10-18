"""Exact prefix counterfactuals from one observed B8 cohort per class.

These are conditional token yields, not separate timed B1..8 experiments.
Only untruncated observations in the retained periodic snapshot are used.
"""
import argparse,collections,gzip,hashlib,json,random,statistics
from pathlib import Path
D=Path(__file__).resolve().parent

def histogram(row,n):
    from riskset_from_ROOT284_histograms_HOST import recover
    recovered=recover({'8':row})['by_budget']['8']
    primary=recovered['primary_path_counts'];alternate=recovered['alternate_path_counts']
    count=row['samples'];out=collections.Counter()
    if n<=4:
        for length,c in enumerate(primary):out[min(length+1,n+1)]+=c
    else:
        both_failed=primary[0]-sum(alternate[1:]);assert both_failed>=0
        out[1]+=both_failed
        for length,c in enumerate(primary[1:],1):out[length+1]+=c
        for length,c in enumerate(alternate[1:],1):out[1+min(length,n-4)]+=c
    assert sum(out.values())==count
    if n==8:assert {k:c for k,c in out.items()if c}=={i:c for i,c in enumerate(row['delivered_histogram'])if c}
    value=sum(k*c for k,c in out.items())
    if n==4:assert value==row['shadow_B4_sum']
    if n==8:assert value==row['delivered_sum']
    return count,value,dict(out)

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--whole463-AR-qualified',action='store_true');args=p.parse_args()
    assert not args.output.exists();raw=args.snapshot.read_bytes();snapshot=json.loads(raw)
    assert snapshot['generation']==5
    grouped={c:[]for c in('tight','normal','loose')}
    for owner,row in snapshot['request_gain_observations']['5'].items():
        observed=row['by_budget'].get('8')
        if not observed:continue
        values=[histogram(observed,n)for n in range(1,9)]
        grouped[row['slo_class']].append(dict(request=owner,samples=observed['samples'],sums=[v[1]for v in values],
            delivered_histograms=[v[2]for v in values]))
    curves={}
    for ordinal,(cls,rows)in enumerate(grouped.items()):
        samples=sum(r['samples']for r in rows);sums=[sum(r['sums'][i]for r in rows)for i in range(8)]
        means=[s/samples for s in sums];rng=random.Random(287+ordinal);draws=[[]for _ in range(8)]
        for _ in range(500):
            selected=[rows[rng.randrange(len(rows))]for _ in rows];total=sum(r['samples']for r in selected)
            for i in range(8):draws[i].append(sum(r['sums'][i]for r in selected)/total)
        low=[sorted(v)[12]for v in draws];high=[sorted(v)[487]for v in draws]
        assert all(1<=g<=5 for g in means)and all(a<=b for a,b in zip(means,means[1:]))
        curves[cls]=dict(B8_observed_requests=len(rows),B8_untruncated_verifications=samples,
            n=list(range(1,9)),mean_delivered=means,request_cluster_bootstrap95=[low,high],
            same_prefix_B8_minus_B4=means[7]-means[3],same_prefix_B8_over_B4=means[7]/means[3],requests=rows)
    result=dict(status='actual_B8_cohort_same_prefix_counterfactual_token_yields',generation=5,
        source_snapshot=str(args.snapshot),source_sha256=hashlib.sha256(raw).hexdigest(),curves=curves,
        whole463_AR_qualified=args.whole463_AR_qualified,all_snapshot_requests=len(snapshot['request_gain_observations']['5']),
        periodic_snapshot_not_final_complete_eventlog=True,snapshot_period_verdict_rows=256,
        measured_B=8,B1_to7='exact same-prefix ancestor-closed counterfactual token counts, not separate runs',
        B8_selection_is_adaptive=True,selection_bias_not_claimed_removed=True,
        confidence_scope='500 request-cluster bootstrap resamples within each observed B8 cohort',
        inference_scope='intrinsic delivered tokens per verification; no latency, throughput or Goodput implication',Goodput=None)
    with args.output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    with args.output.with_suffix('.input.json.gz').open('xb')as f:f.write(gzip.compress(raw,mtime=0))
    print(json.dumps(dict(output=str(args.output),classes={c:{k:v for k,v in row.items()if k!='requests'}for c,row in curves.items()})))

if __name__=='__main__':main()
