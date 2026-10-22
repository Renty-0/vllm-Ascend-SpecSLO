"""Record a terminal209 receipt and full original clocks; no model execution."""
import argparse
import ast
import copy
import math
import re
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics

B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
P=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='small-tail-role-aware-v209-full463-v3'
SOURCE_PINS={'/root/SpecSLO/budget-calibration-20261015/budget-diagnostics/record_v205_terminal_result_ROOT.py': '3a66d16f0358ece807573542bfd5be01c9757fcfb00c5bc9678254b950e1ebe3', '/root/SpecSLO/budget-calibration-20261015/budget-diagnostics/audit_v209_terminal_HOST.py': 'fe8a050d3dce287787017bdf9b0f319bd2577e641e40a7baef7f2df785c0db5b', '/root/SpecSLO/budget-calibration-20261015/budget-diagnostics/ROOT_V209_FUTURE_TERMINAL_HOST_AUDITOR_SOURCE_REVIEW.json': '6177a0ba34dca30261cd92f31b4327b286560cdd40cf20a20d71690e9c784047', '/root/SpecSLO/budget-calibration-20261015/budget-diagnostics/V209_FUTURE_TERMINAL_AUDITOR_SOURCE_READY.json': 'a26ae2629c7fd0e6e86f117e7f3c73adc81a5b31314d7f574ea988e6bf8dc65a', '/root/SpecSLO/budget-calibration-20261015/budget-diagnostics/ROOT_V209_PREPARED_SOURCE_HOST_AUDIT.json': '4f48a31e6dc4e17b13b7b8575569c0542e416c9c82d2658590b57ff0eb8b73c6', '/root/SpecSLO/budget-calibration-20261015/budget-diagnostics/analyze_completed_tree_full463_cost.py': 'd3266647799fc0cf00c49fe98e195ecb85dc446e51ab66c643a5b1d5c3ca0b74', '/root/SpecSLO/budget-calibration-20261015/tree-causal80-small-cohort-role-aware-audit-full-v209/FROZEN_SOURCE209.json': '428fcad26eabc586c9e7991213fd0ebfcf6dbf1a46c4bf58e7bc2cf379ef03ef', '/root/SpecSLO/budget-calibration-20261015/tree-causal80-small-cohort-role-aware-audit-full-v209/CPU_RESULT209.json': '8c1c085cb831ff26a28fda0ba92e908cd730bfdf067b09c25304a5708f5e60d0', '/root/SpecSLO/budget-calibration-20261015/tree-causal80-small-cohort-role-aware-audit-full-v209/audit209.py': '88562baad0dd64f071438b9b307e42059bded28d33a6f3b5eecfeaa0afcee1ad', '/root/SpecSLO/budget-calibration-20261015/tree-causal80-small-cohort-role-aware-audit-full-v209/empty209.py': 'c44c1191632382dbde61b9ce0f26c9544f820f5735e2a45067799f2be165ecf5', '/root/SpecSLO/budget-calibration-20261015/tree-causal80-small-cohort-role-aware-audit-full-v209/trace209.py': '76d9dbfe82c3345c79a686ed6402229090510a2fd87dd4922a8cc97c85cece61'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(path):return dict(path=str(path),sha256=sha(path))
def read(path):
    path=Path(path)
    return json.loads(gzip.decompress(path.read_bytes())if path.suffix=='.gz'else path.read_bytes())
def pct(values,q):
    values=sorted(values);k=(len(values)-1)*q;lo=int(k);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(k-lo)

parser=argparse.ArgumentParser()
parser.add_argument('--terminal-authorized',action='store_true',required=True)
parser.add_argument('--expected-terminal-receipt-sha256',required=True)
parser.add_argument('--session-id',type=int,required=True)
args=parser.parse_args()
assert args.terminal_authorized is True and args.session_id==19465
assert re.fullmatch(r'[0-9a-f]{64}',args.expected_terminal_receipt_sha256) is not None
for source_path,expected in SOURCE_PINS.items():
    assert sha(source_path)==expected,source_path
# FIRST actual-related read below: only after ROOT exact terminal authorization.
receipt=P/f'receipt-{TAG}.json'
assert sha(receipt)==args.expected_terminal_receipt_sha256
r=read(receipt)
assert r['status']=='complete'and r['errors']==[]and r['child']['exit_code']==0
assert r['sources_unchanged']is True and r['owned_remaining']==[]and r['devices_after']['all_idle']is True
assert r['empty_active_only']is r['current_multiowner_eager_math']is r['original_parent173_cap4']is True
assert r['active_windows_requested']==[32]
assert r['action']=='causal80-N9-to15-allB4-empty-active32-full463' and r['tag']==TAG
assert r['new_target_keys']==0 and r['existing_target_keys']==2
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
request_index_by_id={event['request_id']:event['request_index']for event in x['stream_events']}
assert len(request_index_by_id)==463 and set(request_index_by_id.values())==set(range(463))
assert {row['request_id']for row in rows}==set(request_index_by_id)
for label in by_class:
    values=[ttft[request_index_by_id[row['request_id']]]for row in rows if row['slo_class']==label]
    by_class[label]['TTFT_median_ms']=statistics.median(values)
    by_class[label]['TTFT_P90_ms']=pct(values,.9)
    by_class[label]['TTFT_max_ms']=max(values)

# Preserve the exact original charged/256 and publication first-last/255 formulas.
clock_path=D/'analyze_completed_tree_full463_cost.py'
clock_nodes=[copy.deepcopy(node) for node in ast.parse(clock_path.read_text()).body
    if isinstance(node,ast.FunctionDef) and node.name in ('need','clocks')]
assert len(clock_nodes)==2
clock_namespace=dict(math=math,statistics=statistics,CLASSES=('tight','normal','loose'))
exec(compile(ast.fix_missing_locations(ast.Module(body=clock_nodes,type_ignores=[])),
    str(clock_path)+'::ROOT209-original-clocks','exec'),clock_namespace)
original_clock_summary=clock_namespace['clocks'](x)

calls=x['normal_target_calls'];N=Counter(call['N']for call in calls)
owners=sum(call['N']for call in calls)
per_rank={}
for rank in range(4):
    p=A/f'windows/window32/full463/rank{rank}-full463-actual-trace.json.gz'
    value=read(p)
    assert value['cache_released']is True and value['allocator']['final_referenced_pages']==0
    per_rank[str(rank)]=dict(trace=pin(p),allocator=value['allocator'],phase_seconds=value['phase_seconds'])
import sys
sys.path.insert(0,str(P))
from empty209 import audit_trace
cohorts=audit_trace(x)
assert cohorts['no_active_refill']is True and cohorts['all_arrivals_retained']is True
for cohort in cohorts['cohorts']:
    classes={request_index_by_id[row['request_id']]:row['slo_class']for row in rows}
    cohort['class_counts']=dict(Counter(classes[i]for i in cohort['owners']))

tpots=[row['paper_tpot_ms']for row in rows]
result=dict(schema='ROOT.V209.current-full463-result-and-handoff.v1',
    UTC=datetime.now(timezone.utc).isoformat(),status='complete_result_pending_independent_review_not_adopted',
    candidate=str(P),tag=TAG,receipt=pin(receipt),completed=pin(A/'completed.json'),raw=pin(raw),
    session_id=args.session_id,elapsed_seconds=r['elapsed_seconds'],finished_UTC=r['finished_UTC'],
    host_full_seconds=duration,raw_tokens_per_second=463*256/duration,Goodput=good/duration,
    attained=attained,by_class=by_class,
    TPOT=dict(median_ms=statistics.median(tpots),P90_ms=pct(tpots,.9),TPOT_denominator=256,
        clock='original Native charged decode cycle time'),
    TTFT=dict(median_ms=statistics.median(ttft),P90_ms=pct(ttft,.9),
        clock='original worker first-root callback wall minus unchanged rebased original arrival'),
    original_clock_summary=original_clock_summary,
    source_dependencies=SOURCE_PINS,
    future_independent_auditor=pin(D/'audit_v209_terminal_HOST.py'),
    future_independent_auditor_ROOT_review=pin(D/'ROOT_V209_FUTURE_TERMINAL_HOST_AUDITOR_SOURCE_REVIEW.json'),
    original_admission_policy=dict(SLO_AWARE_ADMISSION=False,TARGET_SLO_FIRST=False,source_verified_original197=True),
    current_seven_realN_usage=r['actual_full_audit']['seven_usage'],
    actual_compatibility_classification={k:r['actual_full_audit'][k]for k in('old_padding_behavior_equal','numerical_variant_only')},
    model_output_equivalence_claim=False,
    actual_peak_active=x['active_peak'],normal_target_calls=len(calls),owner_rounds=owners,
    mean_real_N=owners/len(calls),N_histogram=dict(N),
    exact_N16_Q80_causal_calls=sum(c['N']==16 and c['real_query_count']==80 and c['family']=='causal'for c in calls),
    physical_target_Q=sum(c['physical_M']for c in calls),real_target_Q=sum(c['real_query_count']for c in calls),
    prefill_batches=x['decode_counters']['spec_rhythm_prefill_batches'],
    prefill_requests=x['decode_counters']['spec_rhythm_prefill_requests'],
    original_leader_refill_HOST_seconds=x['decode_counters']['spec_rhythm_online_refill_ms']/1000,
    all4=per_rank,output_sha256=hashlib.sha256(json.dumps([row['completion_token_ids']for row in rows],separators=(',',':')).encode()).hexdigest(),
    empty_active_cohort_replay=cohorts,cohort_size_histogram=dict(Counter(len(v['owners'])for v in cohorts['cohorts'])),
    current_multiowner_eager_math=r['actual_full_audit']['current_multiowner_eager_component'],
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
        'Original197 admission0 remains. Only qualified causal uniformB4 N9..15 padding changes M128 to M80. Current same-M checks and explicit cross-M compatibility classification are distinct; changed shape/history is not whole463 canonical equivalence.',
        'Historical Native charged and TP4 engine-publication clocks differ; a formal1.5 claim requires fresh final-nine same-card/common-clock pairing.',
        'Finite same-M component checks do not resolve known canonical own-AR output counterexamples or whole463 equivalence.'
    ])
output=D/'ROOT_V209_V3_CURRENT_FULL463_RESULT_AND_HANDOFF.json'
with output.open('x')as stream:
    json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
print(json.dumps(dict(record=pin(output),Goodput=result['Goodput'],raw=result['raw_tokens_per_second'],
    duration=duration,attained=attained,by_class=by_class,TPOT=result['TPOT'],TTFT=result['TTFT'])))
