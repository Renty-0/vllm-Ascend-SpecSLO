"""Retained causal-chain prediction study; no runtime edit, Torch or device use.

B-specific adaptive observations can compare different prompt populations.
An alternative estimates each conditional prefix acceptance probability,
sharing observed prefixes between B1..4 and censoring untested deeper nodes.
This script evaluates predictions on whole heldout prompt groups; it does
not claim a counterfactual policy Goodput or independent model g experiment.
"""
import collections,datetime,gzip,hashlib,json,math
from pathlib import Path

B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-causal80-small-cohort-role-aware-audit-full-v209/actual/cache-slo1-root-v213-full463-v1'
RAW=P/'windows/window32/full463/rank1-full463-actual-trace.json.gz'
REPORT=P/'rank1-local-diagnostic.json'
OUTPUT=D/'ROOT_V213_GROUP_HELDOUT_PREFIX_HAZARD_GAIN_HOST.json'


def need(value,message):
    if value is not True:raise RuntimeError(message)


def records(trace):
    queues=collections.defaultdict(collections.deque);seen=set()
    for event in trace['stream_events']:
        i=event['request_index']
        if i in seen:queues[i].append(event)
        else:seen.add(i)
    delivered=collections.Counter({i:1 for i in seen});rows=[]
    for call_index,call in enumerate(trace['normal_target_calls']):
        for i,n in zip(call['sequence_ids'][:call['N']],call['budgets'],strict=True):
            event=queues[i].popleft();gain=len(event['token_ids']);remaining=256-delivered[i]
            request=trace['requests'][i]
            row=dict(call=call_index,owner=i,B=n,category=request['slo_class'],
                prompt_group=request['prompt_sha256'],gain=gain,remaining=remaining,
                family=call['family'],physical_M=call['physical_M'],
                nonterminal=remaining>=min(n,4)+1)
            if call['family']=='causal':need(n<=4 and 1<=gain<=n+1,'causal chain gain bounds')
            delivered[i]+=gain;rows.append(row)
    need(not any(queues.values())and len(delivered)==463 and all(v==256 for v in delivered.values()),
         'every original owner target occurrence and256 output reconstructed')
    return rows


def fold(group):
    # Whole identical prompt SHA groups always remain together.
    return 'heldout' if int(group,16)%5==0 else 'train'


def hazard_gain(prior,local,cls,owner,n):
    product=1.;gain=1.
    for depth in range(n):
        alpha,beta=prior[cls,depth];success,failure=local.get((owner,depth),(0,0))
        product*=(alpha+success)/(alpha+beta+success+failure);gain+=product
    return gain


def main():
    need(not OUTPUT.exists(),'fresh private output; previous studies preserved')
    trace=json.loads(gzip.decompress(RAW.read_bytes()));report=json.loads(REPORT.read_text())
    final=report['windows']['traces'][-1]['actual_gain_state']['after']
    base={row['slo_class']:{v['B']:v['g']for v in row['values']}for row in final['base']}
    actual={(row['slo_class'],row['B']):(row['rounds'],row['delivered'])for row in final['observed']}
    rows=records(trace);groups={row['prompt_group']for row in rows}
    train_hazard=collections.defaultdict(lambda:[0,0]);train_specific=collections.defaultdict(lambda:[0,0])
    train_rows=0;heldout_rows=0
    for row in rows:
        if row['family']!='causal' or not row['nonterminal']:continue
        if fold(row['prompt_group'])=='heldout':heldout_rows+=1;continue
        train_rows+=1;cls,n,g=row['category'],row['B'],row['gain'];accepted=g-1
        pair=train_specific[cls,n];pair[0]+=1;pair[1]+=g
        for depth in range(accepted):train_hazard[cls,depth][0]+=1
        if accepted<n:train_hazard[cls,accepted][1]+=1
    priors={};specific={};training=[]
    for cls in ('tight','normal','loose'):
        for depth in range(4):
            success,failure=train_hazard[cls,depth]
            q=(success+1)/(success+failure+2)
            priors[cls,depth]=(12*q,12*(1-q))
            training.append(dict(category=cls,depth=depth+1,accepted=success,rejected=failure,q=q))
        for n in range(1,5):
            count,total=train_specific[cls,n]
            specific[cls,n]=total/count if count else base[cls][n]
    original={};local_hazard={};local_specific={};scores=collections.defaultdict(lambda:[0,0.,0.,0.,0.])
    nonmonotone=collections.Counter();evaluated=0
    for row in rows:
        cls,n,g,owner=row['category'],row['B'],row['gain'],row['owner']
        eligible=row['family']=='causal' and row['nonterminal']
        if eligible and fold(row['prompt_group'])=='heldout':
            count,total=original.get((cls,n),(0,0))
            old=(12*base[cls][n]+total)/(12+count)
            local_count,local_total=local_specific.get((owner,n),(0,0))
            by_budget=(12*specific[cls,n]+local_total)/(12+local_count)
            hazard=hazard_gain(priors,local_hazard,cls,owner,n)
            curves={k:hazard_gain(priors,local_hazard,cls,owner,k)for k in range(1,5)}
            need(all(curves[k+1]>=curves[k]for k in range(1,4)),'intrinsically monotone prefix hazard curve')
            original_curve={}
            for k in range(1,5):
                kcount,ktotal=original.get((cls,k),(0,0))
                original_curve[k]=(12*base[cls][k]+ktotal)/(12+kcount)
            nonmonotone['original_g4_below_g3']+=int(original_curve[4]<original_curve[3])
            nonmonotone['evaluated']+=1;evaluated+=1
            for method,pred in [('recorded_original_class_online',old),('same_train_B_specific_request_local',by_budget),('same_train_prefix_hazard_request_local',hazard)]:
                for category in ('all',cls):
                    value=scores[method,category];error=pred-g
                    value[0]+=1;value[1]+=abs(error);value[2]+=error*error;value[3]+=pred;value[4]+=g
        if row['nonterminal']:
            key=(cls,n);count,total=original.get(key,(0,0));original[key]=(count+1,total+g)
        if eligible:
            count,total=local_specific.get((owner,n),(0,0));local_specific[owner,n]=(count+1,total+g)
            accepted=g-1
            for depth in range(accepted):
                success,failure=local_hazard.get((owner,depth),(0,0));local_hazard[owner,depth]=(success+1,failure)
            if accepted<n:
                success,failure=local_hazard.get((owner,accepted),(0,0));local_hazard[owner,accepted]=(success,failure+1)
    need(original==actual,'complete original class/B observations exactly reconstructed')
    result=[]
    for (method,category),(count,absolute,squared,predicted,actual_total)in sorted(scores.items()):
        result.append(dict(method=method,category=category,rounds=count,MAE=absolute/count,
            RMSE=math.sqrt(squared/count),mean_prediction=predicted/count,mean_actual_gain=actual_total/count))
    row=dict(schema='ROOT213.group-heldout-prefix-hazard-prediction-study.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='completed_HOST_prediction_study_not_new_Native_g',
        inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in(RAW,REPORT,Path(__file__))],
        groups=dict(total=len(groups),train=sum(fold(g)=='train'for g in groups),heldout=sum(fold(g)=='heldout'for g in groups),rule='int(original prompt_sha256,16) mod5 ==0 heldout; all identical prompts remain in one group'),
        causal_nonterminal_rows=dict(train=train_rows,heldout=heldout_rows,evaluated=evaluated),
        prior_strength=12,hyperparameter_search=False,training_conditional_acceptance=training,
        predictor_comparison=result,original_nonmonotone_at_heldout=nonmonotone,
        original_full_observed_counts_exactly_reconstructed=True,
        NPU_used=False,Torch_imported=False,runtime_or_model_changed=False,
        new_policy_installed=False,new_Goodput=None,adopted=False,
        limitations=['All outcomes come from an adaptively selected observational trace; no counterfactual B policy is executed.',
          'Only causal B1..4 nonterminal rows are evaluated; untested deeper nodes are censored, not recorded as rejections.',
          'Original online baseline uses its real12-prior and all earlier classes; both new models train on same other prompt groups, then update only this request past.',
          'No same-prompt group future outcome crosses train/heldout; repeated prompt copies share the split but never share local request history.',
          'Grouped test is distinct from earlier HF/Native177 split; lower error would still require fresh Native g and full performance validation.',
          'B5 side-branch estimation and actual target graph cost are not solved by this prediction study.'])
    OUTPUT.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(path=str(OUTPUT),sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),groups=row['groups'],rounds=row['causal_nonterminal_rows'],all_scores=[v for v in result if v['category']=='all'],new_Goodput=None)))


if __name__=='__main__':main()
