"""Fit class/request conditional gains from actual retained sufficient counts.

Predictions for unsampled budgets are model estimates, not measured outcomes.
No serving code is installed and no NPU operation occurs.
"""
import argparse,copy,gzip,hashlib,json
from pathlib import Path
from riskset_from_ROOT284_histograms_HOST import recover,merge
B=Path('/root/SpecSLO/budget-calibration-20261015');MODEL=B/'tree-high-normal-cap4-native-families-v148/tree-gain-model.json'
CATEGORY={'tight':'coding','normal':'chat','loose':'summarization'}
def priors(curve):
    survival=[];previous=1.;old=1.
    for n in range(1,5):
        value=float(curve[str(n)]);s=min(previous,max(0.,value-old));survival.append(s);old=value;previous=s
    primary=[];previous=1.
    for s in survival:primary.append(s/previous if previous>0 else .5);previous=s
    miss=1-primary[0];bonus=max(0.,float(curve['5'])-float(curve['4']))
    return dict(primary=primary,alternate=[min(1.,bonus/miss)if miss>0 else .5,.5,.5,.5])
def fit(counts,prior,strength=12.):
    assert strength>0
    primary=[(strength*q+r['hits'])/(strength+r['samples'])for q,r in zip(prior['primary'],counts['primary'],strict=True)]
    altcounts=[counts['alternate_first'],*counts['alternate_continuation']]
    alternate=[(strength*q+r['hits'])/(strength+r['samples'])for q,r in zip(prior['alternate'],altcounts,strict=True)]
    curve={'0':0.};product=1.;value=1.
    for n,q in enumerate(primary,1):product*=q;value+=product;curve[str(n)]=value
    survival=1-primary[0];value=curve['4']
    for n,q in enumerate(alternate,5):survival*=q;value+=survival;curve[str(n)]=min(5.,value)
    assert all(0<=p<=1 for p in primary+alternate)and all(curve[str(n)]<=curve[str(n+1)]+1e-12 for n in range(1,8))
    return dict(primary=primary,alternate=alternate,expected_g=curve)
def subtract(total,own):
    out=copy.deepcopy(total)
    for name in('primary','alternate_continuation'):
        for dest,src in zip(out[name],own[name],strict=True):
            for k in('samples','hits'):dest[k]-=src[k];assert dest[k]>=0
    for k in('samples','hits'):out['alternate_first'][k]-=own['alternate_first'][k];assert out['alternate_first'][k]>=0
    return out

def analyze(snapshot,generation):
    requests=snapshot['request_gain_observations'][str(generation)];model=json.loads(MODEL.read_text())
    grouped={c:[]for c in CATEGORY};individual={}
    for owner,row in requests.items():
        counts=recover(row['by_budget']);individual[owner]=(row['slo_class'],counts);grouped[row['slo_class']].append(counts)
    totals={c:merge(rows)for c,rows in grouped.items()}
    prior={c:priors(model['curves'][category])for c,category in CATEGORY.items()}
    classes={c:dict(actual_risk_counts=totals[c],fitted_conditional_model=fit(totals[c],prior[c]),requests_observed=len(grouped[c]))for c in CATEGORY}
    owners={}
    for owner,(cls,counts)in individual.items():
        other_model=fit(subtract(totals[cls],counts),prior[cls])
        prediction=fit(counts,other_model)
        owners[owner]=dict(slo_class=cls,observed_B_means=counts['by_budget'],risk_counts=counts,fitted_conditional_model=prediction)
    return dict(status='actual_histogram_sufficient_statistics_and_model_estimates_only',generation=generation,
        classes=classes,requests=owners,request_prior='other requests in same class plus original12 prior; own data excluded from class prior',
        intrinsic_horizon='only actual observations with at least5 remaining tokens',topology='primary4 plus root alternate up to4',
        unsampled_B_means_not_filled=True,adaptive_selection_can_affect_generalization=True,
        machine_prior_source=str(MODEL),machine_prior_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        performance_result=False,Goodput=None,whole463_AR=False,adopted=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--generation',type=int);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raw=a.snapshot.read_bytes();snapshot=json.loads(raw);generation=a.generation if a.generation is not None else snapshot['generation']
    result=analyze(snapshot,generation);result.update(snapshot_source=str(a.snapshot),snapshot_sha256=hashlib.sha256(raw).hexdigest())
    with a.output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    saved=a.output.with_suffix('.input.json.gz')
    with saved.open('xb')as f:f.write(gzip.compress(raw,mtime=0))
    print(json.dumps(dict(status=result['status'],generation=generation,observed_requests=len(result['requests']),output=str(a.output))))
if __name__=='__main__':main()
