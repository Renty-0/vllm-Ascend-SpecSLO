"""Prepare an unrun first-cohort-only admission candidate from frozen218."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
DONOR=B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'
P=B/'tree-cold-first-cohort-packed-KV-own-AR-full-v219'
assert not P.exists();P.mkdir()
records=[]
for old in sorted(DONOR.iterdir()):
    if old.suffix not in('.py','.sh'):continue
    path=P/old.name.replace('218','219')
    path.write_text(old.read_text().replace(str(DONOR),str(P)).replace('218','219'))
    records.append(dict(source=str(old),source_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),new=str(path)))


def change(name,before,after):
    p=P/name;s=p.read_text();assert s.count(before)==1,(name,before);p.write_text(s.replace(before,after))


(P/'cold219.py').write_bytes((D/'cold_first_cohort_counter_policy_HOST.py').read_bytes())
with(P/'cold219.py').open('a')as stream:stream.write('''

def audit_trace(trace,cache):
    import math
    c=trace['decode_counters']
    need(c['spec_rhythm_cold_first_cohort_minimum_requests']==8
         and c['spec_rhythm_cold_first_cohort_maximum_wait_ms']==1000.,'actual explicit8/1000 policy counters')
    polls=c['spec_rhythm_cold_first_cohort_deferred_polls']
    released=c['spec_rhythm_cold_first_cohort_released_requests']
    waited=c['spec_rhythm_cold_first_cohort_wait_observed_ms']
    need(type(polls)is int and polls>=0 and type(released)is int and 0<released<=32
         and isinstance(waited,(int,float))and math.isfinite(waited)and waited>=0,'actual finite once-initial-cohort counters')
    need(released==len(trace['standalone_prefill_calls'][0]['sequence_ids']),'actual first prefill request count')
    need(released>=8 or waited>=1000.,'original463 trace first-cohort size or timeout release, never fabricated future readiness')
    need(type(cache['cold_first_cohort_source_builds'])is int and cache['cold_first_cohort_source_builds']>0,'actual private builder executed')
    return dict(first_released_requests=released,deferred_polls=polls,first_oldest_wait_ms=waited,
                all_wait_inside_original_generate_clock=True,client_arrival_offsets_modified=False)
''')
change('extension219.py','    Controls,_STATE,_=cached_controls(mapper209,parent)',
       '    Controls,_STATE,_=cached_controls(mapper209,parent)\n    from cold219 import install_builder\n    install_builder(entry,_STATE)')
change('extension219.py','packed_target_KV_preflight=True,',
       'packed_target_KV_preflight=True,cold_first_cohort=True,cold_first_cohort_minimum=8,cold_first_cohort_maximum_wait_ms=1000.,')
change('guard219.py',"        require(r['overlay']['cost_aware_N9to16_budget']is True,'actual cost-aware overlay')",
       "        require(r['overlay']['cost_aware_N9to16_budget']is True,'actual cost-aware overlay')\n        from cold219 import audit_trace as cold_audit_trace\n        cold_audit_trace(trace,cache)\n        require(r['overlay']['cold_first_cohort']is True,'actual explicit cold-first-cohort overlay')")
change('guard219.py','same217 packed preflight plus rank-aware original output-leader comparison artifact and synchronized own-AR load; clients/model/clock unchanged',
       'same218 plus explicit first-cohort server wait minimum8/timeout1000ms; original client arrivals/model/clock formulas unchanged, added waiting remains measured')
change('guard219.py','cost_aware_N9to16_budget=True,packed_target_KV_preflight=True,postfull_independent_FIA_AR=True',
       'cost_aware_N9to16_budget=True,packed_target_KV_preflight=True,cold_first_cohort=True,postfull_independent_FIA_AR=True')
# Extend the existing actual terminal-contract fixture for the new declared fields.
change('host219.py','cost_aware_N9to16_budget=True,packed_target_KV_preflight=True),',
       'cost_aware_N9to16_budget=True,packed_target_KV_preflight=True,cold_first_cohort=True),')
change('host219.py','ROOT219_cached_parent_dispatch=dict(initial_parent_source_checks=1,',
       'ROOT219_cached_parent_dispatch=dict(cold_first_cohort_source_builds=1,initial_parent_source_checks=1,')
change('host219.py',"trace=dict(source_generation=5,normal_target_calls=[{}],decode_counters=dict(spec_rhythm_slo_aware_admission_enabled=1,spec_rhythm_target_slo_first_enabled=0))",
       "trace=dict(source_generation=5,normal_target_calls=[{}],standalone_prefill_calls=[dict(sequence_ids=[0])],decode_counters=dict(spec_rhythm_slo_aware_admission_enabled=1,spec_rhythm_target_slo_first_enabled=0,**dict(__import__('cold219').FIELDS,spec_rhythm_cold_first_cohort_released_requests=1,spec_rhythm_cold_first_cohort_wait_observed_ms=1000.)))")
for row in records:
    old=Path(row['source']).read_text().replace(str(DONOR),str(P)).replace('218','219');new=Path(row['new']).read_text()
    row['new_sha256']=hashlib.sha256(new.encode()).hexdigest()
    row['patch_after_namespace_projection']=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True)))
    if Path(row['new']).suffix=='.py':ast.parse(new)
out=dict(schema='ROOT219.first-cohort-source-projection.v1',donor=str(DONOR),
    donor_freeze_sha256=hashlib.sha256((DONOR/'FREEZE218.json').read_bytes()).hexdigest(),projection=records,
    policy='first empty-service cohort only, minimum8/timeout1000ms, no wait after any request completion',
    source_peer_review=False,NPU_started=False,Goodput=None,adopted=False)
(P/'PROJECTION219.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(status='prepared_unfrozen_unrun',source=str(P),NPU_started=False)))
