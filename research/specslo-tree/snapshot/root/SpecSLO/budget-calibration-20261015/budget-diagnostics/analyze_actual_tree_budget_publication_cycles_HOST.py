"""Join every actual retained-budget ticket to its exact published token event.

Publication intervals include scheduling, draft, target, transport and state
work. They are observational, not isolated kernel costs or causal ablations.
"""
import argparse,bisect,collections,gzip,hashlib,importlib.util,json,statistics,types
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015')

def quantile(values,q):
    values=sorted(values);x=(len(values)-1)*q;i=int(x);j=min(i+1,len(values)-1)
    return values[i]*(1-(x-i))+values[j]*(x-i)
def summary(values):
    return dict(count=len(values),mean=statistics.mean(values),median=statistics.median(values),P90=quantile(values,.9))if values else dict(count=0)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    assert not args.output.exists();raw=args.source.read_bytes();trace=json.loads(gzip.decompress(raw)if args.source.suffix=='.gz'else raw)
    requests={r['request_index']:r for r in trace['requests']};assert len(requests)==463 and all(r['max_tokens']==256 for r in requests.values())
    seen=set();roots=[];events=[]
    for event in trace['stream_events']:
        owner=event['request_index'];assert owner in requests
        if owner not in seen:
            assert len(event['token_ids'])==1;seen.add(owner);roots.append(event)
        else:events.append(event)
    calls=trace['normal_target_calls'];assert len(seen)==463 and sum(c['N']for c in calls)==len(events)
    root_times=sorted(e['elapsed_seconds']for e in roots);published=dict.fromkeys(requests,1)
    cursor=0;previous=None;rows=[];gains=collections.defaultdict(list);terminal=collections.Counter();cycles=collections.defaultdict(list)
    for ordinal,call in enumerate(calls):
        count=call['N'];batch=events[cursor:cursor+count];cursor+=count
        owners=call['sequence_ids'][:count];assert len(owners)==len(set(owners))==count
        assert set(owners)=={e['request_index']for e in batch}
        budgets=dict(zip(owners,call['budgets'],strict=True));delivered=0
        for event in batch:
            owner=event['request_index'];budget=budgets[owner];value=len(event['token_ids']);remaining=256-published[owner]
            assert 1<=value<=min(budget,4)+1 and value<=remaining
            cls=requests[owner]['slo_class'];key=(cls,budget)
            if remaining>=min(budget,4)+1:gains[key].append(value)
            else:terminal[key]+=1
            published[owner]+=value;delivered+=value
            assert bool(event['finished'])==(published[owner]==256)
        end=max(e['elapsed_seconds']for e in batch);begin=min(e['elapsed_seconds']for e in batch)
        assert previous is None or begin>=previous
        crossed_prefill=previous is None or bisect.bisect_right(root_times,end)>bisect.bisect_right(root_times,previous)
        interval_ms=None if crossed_prefill else(end-previous)*1000
        row=dict(ordinal=ordinal,N=count,C=sum(call['budgets']),B5=call['budgets'].count(5),
            budget_histogram=dict(collections.Counter(call['budgets'])),published_tokens=delivered,
            stream_first_seconds=begin,stream_last_seconds=end,publication_interval_ms=interval_ms,
            excluded_first_or_prefill_boundary=crossed_prefill)
        rows.append(row)
        if interval_ms is not None:
            assert interval_ms>0;cycles[(count,row['B5'])].append(row)
        previous=end
    assert cursor==len(events)and all(v==256 for v in published.values())
    model_path=B/'tree-high-normal-cap4-native-families-v148/tree-gain-model.json'
    policy_path=model_path.with_name('budget_policy.py');spec=importlib.util.spec_from_file_location('_HOST_budget_prior_reference',policy_path)
    policy_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy_module)
    policy=policy_module.TreeBudgetPolicy(json.loads(model_path.read_text()))
    gain_rows=[]
    for(cls,budget),values in sorted(gains.items()):
        state=types.SimpleNamespace(slo_class=cls,output_token_budget=256,delivered_tokens=1)
        gain_rows.append(dict(slo_class=cls,B=budget,actual_nonterminal_delivered=summary(values),
            clipped_terminal_rounds_excluded=terminal[(cls,budget)],offline_prior_g=policy.gain(state,budget)))
    cycle_rows=[]
    for(n,branches),selected in sorted(cycles.items()):
        cycle_rows.append(dict(N=n,B5=branches,publication_interval_ms=summary([r['publication_interval_ms']for r in selected]),
            published_tokens_per_call=summary([r['published_tokens']for r in selected])))
    payload=dict(schema='ROOT.actual-budget-and-publication-cycle-HOST-observations.v1',status='all_actual_calls_and_events_joined',
        source=str(args.source),source_sha256=hashlib.sha256(raw).hexdigest(),requests=463,total_tokens=sum(published.values()),
        target_calls=len(calls),nonroot_events=len(events),roots=len(roots),all_owner_ticket_matches_exact=True,
        offline_prior_path=str(model_path),offline_prior_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        gain_rows=gain_rows,cycle_rows=cycle_rows,rounds=rows,
        clock='consecutive actual stream publication endpoints; first and prefill-crossing intervals excluded',
        causal_gain_or_budget_optimum_inferred=False,isolated_target_kernel_cost=False,neural_qualification_inferred=False,
        Goodput_prediction=None,NPU_used=False)
    with args.output.open('x')as f:json.dump(payload,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=payload['status'],target_calls=len(calls),nonroot_events=len(events),gain_rows=gain_rows)))
if __name__=='__main__':main()
