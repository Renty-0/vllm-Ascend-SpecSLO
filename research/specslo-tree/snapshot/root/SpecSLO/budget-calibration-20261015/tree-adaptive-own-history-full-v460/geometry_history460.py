"""Reconstruct per-output physical geometry from real verification positions.

Only request IDs, prompt lengths, root positions, budgets, output counts and
actual M are used. Production token values are not reference inputs.
"""
def need(value,message):
    if value is not True:
        raise RuntimeError('ROOT460 geometry history: '+message)

def build(trace,*,adaptive_required):
    prompts=trace['tokenized_inputs'];requests=trace['requests']
    need(type(adaptive_required)is bool and len(prompts)==len(requests)>0,'explicit protocol and complete requests')
    count=len(prompts);histories=[[]for _ in prompts]
    need(all(row['request_index']==i and row['max_tokens']==256 for i,row in enumerate(requests)),'unchanged explicit output limits')
    for ordinal,row in enumerate(trace['normal_target_calls']):
        n=row['N'];budgets=row['budgets'];owners=row['sequence_ids'][:n];contexts=row['contexts'];m=row['physical_M']
        need(type(n)is int and n==len(budgets)==len(owners)==len(contexts)>0,'all real owner/budget/root records')
        need(len(set(owners))==n and all(type(owner)is int and 0<=owner<count for owner in owners),'distinct real requests')
        need(all(type(b)is int and 1<=b<=5 for b in budgets),'positive original B1to5')
        groups=n+sum(b==5 for b in budgets);queries=n+sum(budgets)
        need(m in(80,120)and groups<=m//5 and queries<=m and row['real_query_count']==queries,'actual entire queries and groups fit M')
        if adaptive_required:
            need(m==(80 if groups<=16 and queries<=80 else 120),'actual declared adaptive geometry policy')
        for owner,context,budget in zip(owners,contexts,budgets,strict=True):
            offset=context-len(prompts[owner])
            need(type(context)is int and type(offset)is int and 0<=offset<255,'actual unpublished next-token frontier')
            if histories[owner]:need(offset>histories[owner][-1]['root_output_index'],'strictly advancing independent request frontier')
            else:need(offset==0,'first target consumes independently prefilled root token')
            histories[owner].append(dict(target_ordinal=ordinal,root_output_index=offset,physical_M=m,budget=budget))
    events=[[]for _ in prompts]
    for event in trace['stream_events']:
        owner=event['request_index']
        need(type(owner)is int and 0<=owner<count,'real published owner')
        tokens=event['token_ids']
        if tokens:events[owner].append(len(tokens))
    schedules=[];intervals=[]
    for owner,history in enumerate(histories):
        need(bool(history),'every real request has target verification history')
        published=events[owner]
        need(len(published)==len(history)+1 and published[0]==1 and sum(published)==256,'one prefill root and all complete verification publications')
        schedule=[0]*256;rows=[]
        for i,record in enumerate(history):
            start=record['root_output_index']+1
            stop=history[i+1]['root_output_index']+1 if i+1<len(history)else 256
            delta=stop-start
            need(1<=delta<=min(record['budget'],4)+1 and delta==published[i+1],'actual frontier advance exactly equals independently observed publication size')
            for position in range(start,stop):
                need(schedule[position]==0,'each generated prediction mapped exactly once')
                schedule[position]=record['physical_M']
            rows.append(dict(record,first_predicted_output_index=start,stop_predicted_output_index=stop,published_count=delta))
        need(schedule[0]==0 and all(m in(80,120)for m in schedule[1:]),'all255 independent AR predictions have one actual physical geometry')
        schedules.append(tuple(schedule));intervals.append(rows)
    return dict(schedules=tuple(schedules),intervals=intervals,requests=count,target_calls=len(trace['normal_target_calls']),
                source='actual target positions and M, checked against nonempty publication counts; production token values excluded',
                adaptive_required=adaptive_required,production_tokens_used_as_reference_inputs=False)
