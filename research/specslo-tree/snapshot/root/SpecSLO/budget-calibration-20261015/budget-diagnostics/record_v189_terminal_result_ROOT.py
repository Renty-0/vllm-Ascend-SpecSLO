"""Record a terminal189 receipt and full original clocks; no model execution."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics

B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
P=B/'tree-slo-admission1-causal80-active32-full-v189'
TAG='slo-admission1-v189-full463-v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(path):return dict(path=str(path),sha256=sha(path))
def read(path):
    path=Path(path)
    return json.loads(gzip.decompress(path.read_bytes())if path.suffix=='.gz'else path.read_bytes())
def pct(values,q):
    values=sorted(values);k=(len(values)-1)*q;lo=int(k);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(k-lo)

parser=argparse.ArgumentParser()
parser.add_argument('--expected-terminal-receipt-sha256',required=True)
args=parser.parse_args()
receipt=P/f'receipt-{TAG}.json'
assert sha(receipt)==args.expected_terminal_receipt_sha256
r=read(receipt)
assert r['status']=='complete'and r['errors']==[]and r['child']['exit_code']==0
assert r['sources_unchanged']is True and r['owned_remaining']==[]and r['devices_after']['all_idle']is True
assert r['SLO_AWARE_ADMISSION']is True and r['TARGET_SLO_FIRST']is False
assert r['prefill_coalesce']==[1,0.0]and r['active_windows_requested']==[32]
assert all(r[k]['exit_code']==0 and r[k]['actual_result']['status']=='pass'for k in('pre_verifier','post_verifier'))
A=P/'actual'/TAG
completed=read(A/'completed.json')
assert completed['audit']==r['actual_full_audit']and r['actual_full_audit']['passed']is True
assert read(A/'worker-exit.json')==dict(closed=True,all_exited=True)
raw=A/'windows/window32/full463/rank1-full463-actual-trace.json.gz'
x=read(raw);rows=x['generate_results']
assert len(rows)==463 and all(len(row['completion_token_ids'])==256 for row in rows)
duration=x['elapsed_seconds']
good=sum(row['paper_slo_goodput_tokens']for row in rows)
attained=sum(row['paper_slo_attained']for row in rows)
assert good==attained*256
by_class={}
for label,count in(('tight',271),('normal',96),('loose',96)):
    selected=[row for row in rows if row['slo_class']==label]
    assert len(selected)==count
    by_class[label]=dict(requests=count,attained=sum(row['paper_slo_attained']for row in selected),
        output_tokens=256*count,good_tokens=sum(row['paper_slo_goodput_tokens']for row in selected),
        TPOT_median_ms=statistics.median(row['paper_tpot_ms']for row in selected),
        TPOT_P90_ms=pct([row['paper_tpot_ms']for row in selected],.9))
first={}
for event in x['stream_events']:
    if 'first_token_wall_seconds160'in event:
        index=event['request_index'];assert index not in first
        first[index]=event['first_token_wall_seconds160']
assert set(first)==set(range(463))
ttft=[1000*(first[index]-x['absolute_arrivals'][index])for index in range(463)]
assert all(value>=0 for value in ttft)
calls=x['normal_target_calls'];N=Counter(call['N']for call in calls)
owners=sum(call['N']for call in calls)
per_rank={}
for rank in range(4):
    p=A/f'windows/window32/full463/rank{rank}-full463-actual-trace.json.gz'
    value=read(p)
    assert value['cache_released']is True and value['allocator']['final_referenced_pages']==0
    per_rank[str(rank)]=dict(trace=pin(p),allocator=value['allocator'],phase_seconds=value['phase_seconds'])
tpots=[row['paper_tpot_ms']for row in rows]
result=dict(schema='ROOT.V189.current-full463-result-and-handoff.v1',
    UTC=datetime.now(timezone.utc).isoformat(),status='complete_result_pending_independent_review_not_adopted',
    candidate=str(P),tag=TAG,receipt=pin(receipt),completed=pin(A/'completed.json'),raw=pin(raw),
    session_id=80092,elapsed_seconds=r['elapsed_seconds'],finished_UTC=r['finished_UTC'],
    host_full_seconds=duration,raw_tokens_per_second=463*256/duration,Goodput=good/duration,
    attained=attained,by_class=by_class,
    TPOT=dict(median_ms=statistics.median(tpots),P90_ms=pct(tpots,.9),TPOT_denominator=256,
        clock='original Native charged decode cycle time'),
    TTFT=dict(median_ms=statistics.median(ttft),P90_ms=pct(ttft,.9),
        clock='original worker first-root callback wall minus unchanged rebased original arrival'),
    actual_peak_active=x['active_peak'],normal_target_calls=len(calls),owner_rounds=owners,
    mean_real_N=owners/len(calls),N_histogram=dict(N),
    exact_N16_Q80_causal_calls=sum(c['N']==16 and c['real_query_count']==80 and c['family']=='causal'for c in calls),
    physical_target_Q=sum(c['physical_M']for c in calls),real_target_Q=sum(c['real_query_count']for c in calls),
    prefill_batches=x['decode_counters']['spec_rhythm_prefill_batches'],
    prefill_requests=x['decode_counters']['spec_rhythm_prefill_requests'],
    original_leader_refill_HOST_seconds=x['decode_counters']['spec_rhythm_online_refill_ms']/1000,
    all4=per_rank,output_sha256=hashlib.sha256(json.dumps([row['completion_token_ids']for row in rows],separators=(',',':')).encode()).hexdigest(),
    original463_arrivals_and256_outputs=True,two_complete_zero65_and_sealed_full463=True,
    sources_unchanged=True,owned_remaining=[],all4_idle=True,
    verifiers={key:r[key]for key in('pre_verifier','post_verifier')},
    original_audit_native_zero_subtractions=r['actual_full_audit']['native_zero_subtractions'],
    independent_terminal_audit_pending=True,independent_terminal_authorized_receipt_SHA=sha(receipt),
    adopted=False,formal_Goodput=False,whole_workload_AR_qualified=False,goal_achieved=False,
    historical_fixed_gamma4_Goodput=501.63560407964184,historical_trueMC2_TP4_Goodput=429.8442987424407,
    provisional_target_1p5x=644.766448113661,
    limitations=[
        'Independent terminal audit and interpretation remain pending; this record does not adopt the candidate.',
        'Historical Native charged and TP4 engine-publication clocks differ; a formal1.5 claim requires fresh final-nine same-card/common-clock pairing.',
        'Finite same-M component checks do not resolve known canonical own-AR output counterexamples or whole463 equivalence.'
    ])
output=D/'ROOT_V189_CURRENT_FULL463_RESULT_AND_HANDOFF.json'
with output.open('x')as stream:
    json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
print(json.dumps(dict(record=pin(output),Goodput=result['Goodput'],raw=result['raw_tokens_per_second'],
    duration=duration,attained=attained,by_class=by_class,TPOT=result['TPOT'],TTFT=result['TTFT'])))
