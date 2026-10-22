"""Only new external V148 HOST bridge/schema regressions; never candidate suites or devices."""
import copy
import gzip
import importlib
import json
from pathlib import Path
import sys
import unittest

D=Path(__file__).resolve().parent
sys.path.insert(0,str(D))
import analyze_v148_actual_full_cap4 as audit
import analyze_v144_actual_variable_B4 as component


class V148ExternalHOST(unittest.TestCase):
    def test_exact21_counter_bridge_and_zero_replay_negative_cases(self):
        before=dict.fromkeys(audit.NATIVE_COUNTERS,10)
        pair=dict(before=before,after=dict(before),delta=dict.fromkeys(before,0))
        self.assertEqual(audit.component_native_pair(pair,before,zero=True),before)
        changed=copy.deepcopy(pair);changed['before']['aclgraph_captures']+=1
        with self.assertRaises(ValueError):audit.component_native_pair(changed,before)
        changed=copy.deepcopy(pair);changed['after']['aclgraph_captures']+=1;changed['delta']['aclgraph_captures']=1
        self.assertEqual(audit.component_native_pair(changed,before)['aclgraph_captures'],11)
        with self.assertRaises(ValueError):audit.component_native_pair(changed,before,zero=True)
        changed=copy.deepcopy(pair);changed['after']['aclgraph_shape_fallbacks']+=1;changed['delta']['aclgraph_shape_fallbacks']=1
        with self.assertRaises(ValueError):audit.component_native_pair(changed,before)
        with self.assertRaises(ValueError):audit.component_native_pair(dict(before={},after={},delta={}),{})

    def test_inherited_greedy_and_new_hidden_natural_flag_sequences(self):
        flags={'greedy':True}
        for phase in ('first','changed','restored'):
            audit.component_target_flags(dict(phase=phase,key='greedy',runtime_validated_before=True,
                runtime_validated_after=True),'greedy',flags)
        for phase,before,after in [('first',False,False),('changed',False,True),('restored',True,True)]:
            audit.component_target_flags(dict(phase=phase,key='hidden',runtime_validated_before=before,
                runtime_validated_after=after),'hidden',flags)
        self.assertEqual(flags,dict(greedy=True,hidden=True))
        with self.assertRaises(ValueError):audit.component_target_flags(dict(phase='first',key='greedy',
            runtime_validated_before=False,runtime_validated_after=False),'greedy',flags)
        with self.assertRaises(ValueError):audit.component_target_flags(dict(phase='first',key='new',
            runtime_validated_before=True,runtime_validated_after=True),'new',{})

    def test_real_retained_V143_native_dictionaries_not_logical_report_labels(self):
        root=audit.B/'tree-standalone-prefill-control-v143/local-output/standalone-prefill-v143-one-full463-v1'
        for rank in range(4):
            report=json.loads((root/f'rank{rank}-local-diagnostic.json').read_text())
            chain=json.loads(gzip.decompress(Path(report['mixed_same_M_qualification']['artifact']['path']).read_bytes()))
            before=chain['actual_inventory']['after']
            counts=audit.component_inventory_keys(before,rank,after=False)
            self.assertEqual(counts,dict(entries=0,draft_entries=60,target_entries=54) if rank==0
                else dict(entries=0,draft_entries=0,target_entries=247))
            after=copy.deepcopy(before)
            if rank:
                for m,s,_ in component.GEOMETRIES:after['target_entries'].append(repr(component.target_key(m,s,True)))
                after['target_entries'].sort()
            audit.component_inventory_keys(after,rank,after=True)
            wrong=dict(entries=list(before['target_entries'][:201]),draft_entries=[],target_entries=list(before['target_entries'][201:]))
            with self.assertRaises(ValueError):audit.component_inventory_keys(wrong,rank,after=False)

    def test_new_full_component_exact_pre_save_stage_prefix(self):
        labels=audit.component_boundary_labels(component)
        events=[dict(sequence=i,label=label,ranks=[dict(rank=r,sequence=i,label=label,error=None)
            for r in range(4)]) for i,label in enumerate(labels)]
        self.assertEqual(audit.component_boundaries(events,component),1859)
        with self.assertRaises(ValueError):audit.component_boundaries(events[:-1],component)
        wrong=copy.deepcopy(events);wrong[-1]['ranks'][2]['error']='injected HOST negative'
        with self.assertRaises(ValueError):audit.component_boundaries(wrong,component)
        wrong=copy.deepcopy(events);wrong[4]['label']='minimal-construction'
        with self.assertRaises(ValueError):audit.component_boundaries(wrong,component)

    def test_actual_retained_causal_verdict_and_full8_encoded_reconstruction(self):
        root=audit.B/'tree-variable-B4-local-qualification-v144/actual/variable-B4-local-v144-1'
        draft=json.loads(gzip.decompress((root/'rank0-pass0-case0-cycle1.json.gz').read_bytes()))
        target=json.loads(gzip.decompress((root/'rank1-pass0-case0-cycle1.json.gz').read_bytes()))
        budgets=component.VECTORS[0]
        component.draft_ABI(draft['draft'],budgets)
        query=target['target']['greedy'][2]['full_physical_tokens'][:32+sum(budgets)]
        emitted,accepted=component.causal_verdict(draft['rows'],query,budgets)
        self.assertEqual((emitted,accepted),(draft['commit']['committed_tokens'],draft['commit']['accepted_paths']))
        wrong=copy.deepcopy(draft['draft']);wrong['encoded_sha256']='0'*64
        with self.assertRaises(ValueError):component.draft_ABI(wrong,budgets)
        with self.assertRaises(ValueError):component.causal_verdict(draft['rows'],query[:-1],budgets)

    def test_live_highN_cap4_admission_accepts4_and_rejects5(self):
        base=dict(available=True,rule=audit.RULE,generation=7,call_count=0,ages=[],calls=[],overflow_calls=0,
            raw_N_histogram=[0]*65,admitted_N_histogram=[0]*33,positive_grant_N_histogram=[0]*33)
        after=copy.deepcopy(base);after.update(call_count=1,ages=[dict(owner=32,age=1)],overflow_calls=1,
            Native_home_or_KV_mapping_changed=False,next_cycle_service_guaranteed=False)
        after['raw_N_histogram'][33]=after['admitted_N_histogram'][32]=after['positive_grant_N_histogram'][32]=1
        event=dict(call=0,raw_normal=list(range(33)),post_window_eager=[],admitted_normal=list(range(32)),
            explicit_active_batch_size=33,classes=['tight']*33,ages_before=[0]*33,
            positive_normal_grants=[dict(owner=i,B=b) for i,b in enumerate(component.VECTORS[0])],
            cap_deferred_normal=[32],zero_grant_deferred_normal=[],all_deferred_normal=[32],ages_after=[0]*32+[1],
            raw_N=33,admitted_N=32,positive_grant_N=32,eager_N=0,overflow=True,
            actual_work_normal_row_limit=32,actual_work_combined_row_limit=64)
        after['calls']=[event]
        policy={f'draft_normal_N{n}_shape_calls':int(n==32) for n in range(33)}
        policy.update({f'actual_draft_normal_N{n}_work_calls':int(n==32) for n in range(33)})
        policy['actual_draft_normal_rows']=32
        proof=audit.audit_admission(dict(before=base,after=after),generation=7,count=33,
            classes=dict.fromkeys(range(33),'tight'),policy=policy)
        self.assertEqual(proof['overflow_calls'],1)
        wrong=copy.deepcopy(after);wrong['calls'][0]['positive_normal_grants'][0]['B']=5
        with self.assertRaises(ValueError):audit.audit_admission(dict(before=base,after=wrong),generation=7,count=33,
            classes=dict.fromkeys(range(33),'tight'),policy=policy)


if __name__=='__main__':unittest.main()
