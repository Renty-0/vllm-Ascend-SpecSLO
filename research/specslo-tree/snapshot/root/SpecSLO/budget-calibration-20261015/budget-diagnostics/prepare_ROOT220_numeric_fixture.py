"""Prepare bounded diagnostics at three real218 first differences, no backend."""
from pathlib import Path
import gzip
import hashlib
import json
from datetime import datetime, timezone

B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-same-prefix-logits-KV-numeric-forensics-v220'
ACTUAL=B/'tree-causal80-small-cohort-role-aware-audit-full-v209/actual/packed-KV-rolefix-root-v218-full463-AR-v1'
assert not P.exists();P.mkdir()


def read(path):
    data=path.read_bytes();return json.loads(gzip.decompress(data)if path.suffix=='.gz'else data)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


trace_path=ACTUAL/'windows/window32/full463/rank1-full463-actual-trace.json.gz'
location_path=D/'ROOT_V218_ALL_OWN_AR_FIRST_DIFFERENCE_LOCATIONS.json'
trace=read(trace_path);locations=read(location_path)
requests=trace['requests'];prompts=trace['tokenized_inputs']
candidate={r['request_id']:r['completion_token_ids']for r in trace['generate_results']}
by_id={r['request_id']:i for i,r in enumerate(requests)}
references={};cohorts={};input_files={str(trace_path):sha(trace_path),str(location_path):sha(location_path)}
for path in sorted((ACTUAL/'postfull-independent-FIA-AR').glob('rank1-cohort*.json.gz')):
    row=read(path);input_files[str(path)]=sha(path)
    for owner,rid,tokens in zip(row['original_owner_indices'],row['request_ids'],row['reference_completions'],strict=True):
        references[owner]=tokens;cohorts[owner]=row['original_owner_indices']
assert len(references)==463
seen=set();normal=[]
for event in trace['stream_events']:
    owner=event['request_index']
    if owner not in seen:seen.add(owner)
    else:normal.append(event)
cursor=0;before=[1]*463;call_data={}
for ordinal,call in enumerate(trace['normal_target_calls']):
    events=normal[cursor:cursor+call['N']];cursor+=call['N']
    owners=call['sequence_ids'][:call['N']]
    assert [r['request_index']for r in events]==owners
    call_data[ordinal]=(dict(call),{i:before[i]for i in owners},events)
    for owner,event in zip(owners,events,strict=True):before[owner]+=len(event['token_ids'])
assert cursor==len(normal)
different={r['request_id']:r for r in locations['first_differences']}
cases=[]


def origin(owner,count,kind):
    tokens=candidate[requests[owner]['request_id']]if kind=='candidate'else references[owner]
    assert count>0 and count<256
    return dict(original_owner=owner,request_id=requests[owner]['request_id'],history_source=kind,
                prefix=list(prompts[owner])+list(tokens[:count-1]),root=tokens[count-1])


for suffix,kind in [('000000','full-vs-causal'),('000129','within-chain-vs-next-root'),('000034','batch-composition')]:
    rid='rps4p0-seed0-'+suffix;difference=different[rid]
    owner=by_id[rid];position=difference['first_difference_zero_based'];loc=difference['location']
    assert candidate[rid][:position]==references[owner][:position]
    call,begins,events=call_data[loc['call_ordinal']]
    assert loc['budget']in(4,5)and call['N']in(1,16)
    rows=[origin(i,begins[i],'candidate')for i in call['sequence_ids'][:call['N']]]
    own_row=loc['owner_row'];prefix=len(rows[own_row]['prefix']);offset=loc['delivered_chunk_offset']
    assert prefix==loc['context']and begins[owner]+offset==position
    frames=[]
    if kind=='full-vs-causal':
        assert call['N']==1 and offset==0 and position==85
        for name,budget,M,S in [('FULL40',5,40,8),('CAUSAL80',4,80,16),('FULL40_REPEAT',5,40,8)]:
            frames.append(dict(name=name,ids=[0],budgets=[budget],shifts=[0],roots=[rows[0]['root']],
                futures=[[0]*budget],M=M,S=S,affected_row=0,query_offset=0))
    elif kind=='within-chain-vs-next-root':
        assert offset==1 and call['physical_M']==80 and call['budgets']==[4]*16
        futures=[]
        for row in rows:
            i=row['original_owner'];count=begins[i]
            values=list(candidate[row['request_id']][count:count+4])
            assert len(values)==4
            if i==owner:values=[values[0],0,0,0]
            futures.append(values)
        original_roots=[r['root']for r in rows]
        next_roots=[candidate[r['request_id']][begins[r['original_owner']]]for r in rows]
        for name,roots,future,shift,q in [('QUERY1',original_roots,futures,0,1),
            ('ROOT0_FUTURE_ZERO',original_roots,[[0]*4 for _ in rows],0,0),
            ('NEXT_ROOT',next_roots,[[0]*4 for _ in rows],1,0),
            ('QUERY1_REPEAT',original_roots,futures,0,1)]:
            frames.append(dict(name=name,ids=list(range(16)),budgets=[4]*16,shifts=[shift]*16,
                roots=roots,futures=future,M=80,S=16,affected_row=own_row,query_offset=q))
    else:
        assert offset==0 and call['physical_M']==80 and call['budgets']==[4]*16
        group=cohorts[owner];index=group.index(owner);begin=index//16*16;reference_group=group[begin:begin+16]
        assert len(reference_group)==16
        b_ids=[]
        for i in reference_group:
            if i==owner:b_ids.append(own_row)
            else:b_ids.append(len(rows));rows.append(origin(i,position,'reference'))
        for name,ids,row_index in [('HOME_GROUP',list(range(16)),own_row),
            ('REFERENCE_CONTIGUOUS_GROUP',b_ids,reference_group.index(owner)),
            ('HOME_GROUP_REPEAT',list(range(16)),own_row)]:
            frames.append(dict(name=name,ids=ids,budgets=[4]*16,shifts=[0]*16,
                roots=[rows[i]['root']for i in ids],futures=[[0]*4 for _ in ids],M=80,S=16,
                affected_row=row_index,query_offset=0))
    pages=sum((len(r['prefix'])+10+127)//128 for r in rows)+64
    assert pages<=544 and all(len(r['prefix'])+10<4096 for r in rows)
    cases.append(dict(name=kind,request_id=rid,original_owner=owner,first_difference=position,
        actual_candidate_token=candidate[rid][position],actual_reference_token=references[owner][position],
        actual_location=loc,affected_cache_owner=own_row,affected_initial_prefix=prefix,
        rows=rows,frames=frames,conservative_unique_page_upper=pages,
        affected_prefix_identical_before_real_first_difference=True,
        diagnostic_teacher_forcing_of_recorded_COMMON_history=True,
        not_an_independent_AR_qualification=True))
fixture=dict(schema='ROOT220.same-prefix-numeric-diagnostic-fixture.v1',UTC=datetime.now(timezone.utc).isoformat(),
    inputs=input_files,cases=cases,model_selection_or_sampler_unchanged=True,
    purpose='Isolate layout/query-position/grouping effects using the same shared affected-prefix KV; not a performance or independent AR qualification',
    NPU_used=False,formal_Goodput=False,adopted=False)
(P/'FIXTURE220.json').write_text(json.dumps(fixture,indent=2)+'\n')
print(json.dumps(dict(directory=str(P),cases=[dict(name=c['name'],rows=len(c['rows']),pages=c['conservative_unique_page_upper'],frames=len(c['frames']))for c in cases],NPU_used=False)))
