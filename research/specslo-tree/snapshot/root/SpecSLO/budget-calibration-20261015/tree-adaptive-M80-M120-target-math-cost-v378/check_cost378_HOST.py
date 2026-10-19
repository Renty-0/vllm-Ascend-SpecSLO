"""Actual six-arm order on HOST doubles plus independent auditor corruptions."""
import copy
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace

P=Path(__file__).parent;B=P.parent
sys.path[:0]=[str(P),str(B/'tree-adaptive-M80-M120-target-host-v377'),
    str(B/'tree-static-q5-M120-node-math-v255'),str(B/'tree-canonical-contiguous-full463-own-AR-v235')]


def main():
    import benchmark378,audit378,oracle255
    from geometry377 import choose
    def need(x):
        if x is not True:raise RuntimeError('ROOT378 HOST cost regression')
    def rejects(fn):
        try:fn()
        except(RuntimeError,ValueError,KeyError):return
        raise RuntimeError('ROOT378 HOST corrupted panel accepted')
    keys=('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays',
          'aclgraph_failed_captures','aclgraph_capacity_fallbacks','aclgraph_shape_fallbacks')
    fake_torch=ModuleType('torch')
    old_torch=sys.modules.get('torch');old_prepare=oracle255.prepare
    need(old_torch is None)
    sys.modules['torch']=fake_torch
    oracle255.prepare=lambda case,prefixes:([object()for b in case['budgets']],[[0]*b for b in case['budgets']])
    try:
        engine=SimpleNamespace(is_draft=True,rank=0,_branch_tail_state=SimpleNamespace(adaptive377_enabled=True),
            graph_metrics=lambda:{k:0 for k in keys})
        bank=dict(prefixes=[[0]*127 for _ in range(24)])
        draft=benchmark378.run(engine,None,lambda label,fn:fn(),bank)
    finally:
        oracle255.prepare=old_prepare
        sys.modules.pop('torch',None)
    need(len(draft['pairs'])==20 and len(draft['arms'])==6)
    first={a['name']:{m:0 for m in draft['modes']}for a in draft['arms']}
    for pair in draft['pairs']:
        for sample in pair['samples']:first[sample['arm']][sample['mode_order'][0]]+=1
    need(all(n==10 for bymode in first.values()for n in bymode.values()))
    need(audit378.panel(dict(paired_physical_cost=draft,actual_cost_calls=[],all72_prefixes_after_cost=None),
         bank['prefixes'],0)==[])
    # This synthetic target fixture validates the accounting protocol only.
    # Its invented durations are deliberately never retained as performance.
    target=copy.deepcopy(draft);target.update(rank=1,target_math_on_rank=True)
    ordered=[(arm,mode)for arm in target['arms']for mode in target['modes']]
    for pair in target['pairs']:
        for sample in pair['samples']:
            sample['milliseconds']={m:1.0 for m in target['modes']}
            arm=next(a for a in target['arms']if a['name']==sample['arm'])
            ordered.extend((arm,mode)for mode in sample['mode_order'])
    calls=[]
    for arm,mode in ordered:
        m=choose(arm['budgets'],enabled=mode=='adaptive80_120');s=m//5
        visible=[127+min(b,4)+1 for b in arm['budgets']]
        calls.append(dict(N=arm['N'],real_queries=arm['N']+sum(arm['budgets']),M=m,attention_segments=s,
            actual_seq_lengths_q=list(range(5,m+1,5)),sequence_lens=[512]*s,
            mask_shape=[s,1,11 if arm['budget']==5 else 10,4096],copy_enabled=arm['budget']==5,
            used_aclgraph=True,execution_mode='replay',fallback_reason=None,
            primary_side_groups=arm['N']+arm['budgets'].count(5),adaptive377_enabled=mode=='adaptive80_120',
            uniform_KV_bucket377=dict(enabled=True,extent=512,original_sequence_lens=visible)))
    fixture=dict(paired_physical_cost=target,actual_cost_calls=calls,all72_prefixes_after_cost=True)
    need(len(audit378.panel(fixture,bank['prefixes'],1))==6)
    bad=copy.deepcopy(fixture);bad['actual_cost_calls'].pop();rejects(lambda:audit378.panel(bad,bank['prefixes'],1))
    bad=copy.deepcopy(fixture);bad['actual_cost_calls'][1]['M']=120;rejects(lambda:audit378.panel(bad,bank['prefixes'],1))
    bad=copy.deepcopy(fixture);bad['paired_physical_cost']['pairs'][1]['samples'][0]['mode_order'].reverse();rejects(lambda:audit378.panel(bad,bank['prefixes'],1))
    bad=copy.deepcopy(fixture);bad['paired_physical_cost']['pairs'][0]['samples'][0]['milliseconds']['fixed120']=float('nan');rejects(lambda:audit378.panel(bad,bank['prefixes'],1))
    bad=copy.deepcopy(fixture);bad['all72_prefixes_after_cost']=False;rejects(lambda:audit378.panel(bad,bank['prefixes'],1))
    bad=copy.deepcopy(fixture);bad['paired_physical_cost']['graph_counters_after']['aclgraph_captures']=1;rejects(lambda:audit378.panel(bad,bank['prefixes'],1))
    need('torch'not in sys.modules and 'torch_npu'not in sys.modules)
    value=dict(status='PASS_HOST378_ACTUAL_SIX_ARM_ORDER_AND_AUDIT_CORRUPTIONS',NPU_used=False,
        actual_original_timing_loop_exercised_on_draft_HOST_double=True,each_arm_mode_first_counts=first,
        target_auditor_fixture_is_synthetic=True,synthetic_timings_saved_as_results=False,
        target_calls_each_bank=252,corruption_controls=6,model_qualified=False,Raw=None,Goodput=None)
    (P/'HOST378_COST_AND_AUDIT_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':main()
