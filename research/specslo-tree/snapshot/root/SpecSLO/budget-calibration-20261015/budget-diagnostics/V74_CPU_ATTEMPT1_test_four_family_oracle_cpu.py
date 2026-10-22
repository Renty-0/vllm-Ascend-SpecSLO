"""Focused real-metadata and fail-closed CPU contracts, never NPU proof."""
import copy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


def prepared(case):
    import four_family_cpu_fixture as fixture,four_family_scope as scope,four_family_target as target
    from vllm_ascend.spec_decode.pearl.tree import pack_selected_tree_plan
    e=fixture.engine(False);prefixes=[127+i for i in range(16)]
    plans=scope.plans(prefixes,case['budgets']);chosen=[]
    for i,B in enumerate(case['budgets']):
        probability=[.95,.9,.85,.8,.1,.1,.1,.1];probability[4+i%4]=.7
        chosen.append(scope.selections(probability,B))
    selected=[pack_selected_tree_plan(p,nodes) for p,nodes in zip(plans,chosen)]
    rows=[[int(j%4) for j in nodes] for nodes in chosen]
    padded=scope.pad(e,selected,[2]*16,rows,list(range(16)),case)
    return e,padded,target.prepare(e,padded,case)


def original_entry(call):
    import torch
    from vllm_ascend.spec_decode.pearl.native_graph import NativeTargetACLGraphEntry
    x,pos,meta=call;M=len(x);tree=bool(meta.tree_attention)
    mask=meta.tree_attention_mask if tree else meta.attention_mask
    tasks=[NS(query=NS(shape=(M,1)),output=NS(shape=(M,1)),block_table=meta.request_block_tables,
              block_size=128,tree_attention=tree,attention_mask=mask,handle=object(),event=object()) for _ in range(64)]
    return NativeTargetACLGraphEntry(input_ids=(x,),positions=(pos,),slot_mappings=(meta.slot_mapping,),
        context_lens=(meta.context_lens,),block_tables=(meta.block_tables,),request_block_tables=(meta.request_block_tables,),
        attention_masks=(meta.attention_mask,),actual_seq_lengths_q=(meta.actual_seq_lengths_q,),
        sequence_lens=(meta.sequence_lens,),graph=NS(),outputs=(torch.zeros(M,2),),tasks=tasks,
        tasks_per_step=64,runtime_validated=True,tree_attention_masks=(meta.tree_attention_mask,),tree_attention_modes=(tree,))


class FourFamilyOracleCPU(unittest.TestCase):
    def test_real_native_four_family_metadata_preserves_every_query_and_scratch(self):
        import torch
        import four_family_scope as scope,source_epoch as epoch
        for case in scope.cases():
            e,padded,call=prepared(case);x,pos,meta=call
            self.assertEqual((len(x),len(meta.sequence_lens)),(case['M'],case['S']))
            q=[p.candidate_budget+1 for p in padded[0]]
            self.assertEqual(q[:16],[b+1 for b in case['budgets']])
            self.assertEqual(q[16:],list(case['scratch_Q']))
            self.assertEqual(sum(q[:16]),case['Q']);self.assertEqual(sum(q),case['M'])
            self.assertEqual(bool(meta.tree_attention),case['family']=='full9')
            if case['family']=='full9':self.assertEqual(tuple(meta.tree_attention_mask.shape),(16,1,9,4096))
            else:self.assertEqual((case['M']-case['Q'],len(q[16:])),(48,22))
            e.model(*call);kv=epoch.query_KV(e,meta)
            self.assertTrue(all(v.shape[0]==case['M'] and bool(torch.isfinite(v).all()) for v in kv))
            self.assertEqual(len(set(meta.slot_mapping.tolist())),case['M'])

    def test_original_ancestry_guard_is_retained_for_mixed_budget_target(self):
        from dataclasses import replace
        import four_family_scope as scope,four_family_target as target
        e,padded,_=prepared(scope.cases()[0]);pp=list(padded[0]);position=pp[0].positions.clone();position[-1]+=1
        pp[0]=replace(pp[0],positions=position)
        with self.assertRaisesRegex(RuntimeError,'independent ancestry'):
            target.prepare(e,(pp,*padded[1:]),scope.cases()[0])

    def test_plain_host_descriptor_matches_real_native_keys_without_builder(self):
        import four_family_scope as scope,four_family_mc2 as mc2
        for case in scope.cases():
            e,_,call=prepared(case)
            descriptor=mc2.descriptor(e,case)
            self.assertIsInstance(descriptor[0],range)
            self.assertEqual(mc2.keys(call,4),mc2.keys(descriptor,4))
            key=mc2.keys(call,4)['greedy']
            self.assertEqual(key[1],(case['M'],))
            self.assertIn('tree-fia-full' if case['family']=='full9' else '|fia:',key[0])

    def test_all11_original_MC2_counter_fields_types_and_cold_replay_are_strict(self):
        import four_family_mc2 as mc2
        from vllm_ascend.spec_decode.pearl.mc2 import _MC2_DISPATCH_COUNTER_NAMES
        self.assertEqual(mc2.COUNTER_FIELDS,_MC2_DISPATCH_COUNTER_NAMES)
        zero=dict.fromkeys(mc2.COUNTER_FIELDS,0)
        self.assertFalse(mc2.raw_dispatch(zero,zero,require_native_success=True)['passed'])
        self.assertTrue(mc2.raw_dispatch(zero,zero,require_native_success=False)['passed'])
        for field in mc2.COUNTER_FIELDS:
            missing=dict(zero);missing.pop(field)
            bad=mc2.raw_dispatch(missing,missing,require_native_success=False)
            self.assertFalse(bad['passed']);self.assertEqual(bad['before'],missing)
        for field in ('fallback','exception'):
            after=dict(zero);after[field]=1
            self.assertFalse(mc2.raw_dispatch(zero,after,require_native_success=False)['passed'])
        after=dict(zero);after['native_epilogue_success']=128
        self.assertTrue(mc2.raw_dispatch(zero,after,require_native_success=True)['passed'])
        boolean=dict(zero);boolean['fused_success']=True
        self.assertFalse(mc2.raw_dispatch(boolean,boolean,require_native_success=False)['passed'])
        self.assertFalse(mc2.raw_dispatch(after,zero,require_native_success=False)['passed'])

    def test_real_native_entry_schema_and64_independent_task_handles_are_checked(self):
        import four_family_scope as scope,four_family_mc2 as mc2
        for case in scope.cases():
            _,_,call=prepared(case);entry=original_entry(call)
            self.assertTrue(mc2.entry_state(entry,call)['valid'])
            entry.runtime_validated=False;self.assertFalse(mc2.entry_state(entry,call)['valid'])
            entry.runtime_validated=True;entry.tasks[-1].handle=entry.tasks[0].handle
            self.assertFalse(mc2.entry_state(entry,call)['valid'])

    def test_plain_restoration_rejects_first_or_next_KV_ledger_and_confidence_drift(self):
        import torch
        import four_family_oracle as oracle
        e=NS(is_draft=True,graph_runner=NS(_target_outputs_match=lambda a,b:
            len(a)==len(b) and all(torch.equal(x,y) for x,y in zip(a,b))))
        output=dict(draft_token_ids=[[1]*8],root_token_ids=[2],node_confidences=[torch.tensor([.5]*8)],
            draft_confidence=torch.tensor([.5]),query_count=72,execution_mode='CPU-fake',model_stages=4)
        snap=dict(output=output,ledger={0:dict(safe_before=2,tokens={2:2})},KV=(torch.ones(2,2),),source_prefix=['exact-source-hash'])
        baseline=dict(plain_path_signature={'CPU_fake':True})
        snapshots=dict(first=snap,next=copy.deepcopy(snap),original_source_prefix=['source'])
        self.assertTrue(oracle.compare_plain_restore(e,baseline,copy.deepcopy(baseline),snapshots,copy.deepcopy(snapshots))['passed'])
        for kind in ('first','next'):
            bad=copy.deepcopy(snapshots);bad[kind]['KV'][0][0,0]+=1
            with self.assertRaisesRegex(RuntimeError,kind):oracle.compare_plain_restore(e,baseline,baseline,snapshots,bad)
            bad=copy.deepcopy(snapshots);bad[kind]['ledger'][0]['tokens'][2]=3
            with self.assertRaisesRegex(RuntimeError,kind):oracle.compare_plain_restore(e,baseline,baseline,snapshots,bad)
            bad=copy.deepcopy(snapshots);bad[kind]['output']['node_confidences'][0][0]+=.1
            with self.assertRaisesRegex(RuntimeError,kind):oracle.compare_plain_restore(e,baseline,baseline,snapshots,bad)

    def test_cost_entry_never_requests_bridge_or_scoped_qualification(self):
        import four_family_oracle as oracle
        marker=object();engine=object();native=object();case={};source={};callbacks=object()
        with patch.object(oracle,'_round',return_value=marker) as run:
            self.assertIs(oracle.run_plain(engine,native,case,source,callbacks=callbacks),marker)
            run.assert_called_once_with(engine,native,case,source,callbacks,qualification=False)
        source=dict(contexts=[dict(token_ids=[1,2])]*16,repairs=[1]*8+[0]*8)
        rebuilt=dict(prefixes=[1]*16,roots=[2]*16,ids=list(range(16)),catchups=[[1,2]]*8+[[2]]*8)
        independent_target_KV=object()
        baseline=dict(_functional_snapshots={'target_KV':independent_target_KV},plain_path_signature={'true-method':1})
        restored=dict(_functional_snapshots={'CPU-restore':True},plain_path_signature={'true-method':1},target_commit_oracle={'passed':True})
        callback=NS(stage=lambda engine,label,function:function())
        with patch.object(oracle,'_round',side_effect=[baseline,restored]) as run,patch.object(
                oracle.epoch,'rebuild',return_value=rebuilt) as rebuild,patch.object(
                oracle,'compare_plain_restore',return_value={'passed':True}) as compare:
            result=oracle.qualify(engine,native,case,source,callbacks=callback)
            rebuild.assert_called_once_with(engine,source['contexts'],source['repairs'])
            self.assertTrue(run.call_args_list[1].kwargs['bridge'])
            self.assertFalse(run.call_args_list[1].kwargs['qualification'])
            self.assertIs(run.call_args_list[1].kwargs['bridge_reference_KV'],independent_target_KV)
            self.assertTrue(result['plain_path_checksum_bridge_passed'])
            self.assertTrue(result['candidate_actual_state_restored_after_independent_reference'])
            self.assertNotIn('_functional_snapshots',result);compare.assert_called_once()

    def test_scoped_witness_actual_raw26_PAD_ownership_and_query_only_prefix_bounds(self):
        import four_family_cpu_fixture as fixture,four_family_native_witness as witness
        from vllm_ascend.spec_decode.pearl.native_graph import NativeACLGraphRunner
        import torch
        e=fixture.engine(True);e.model.layers=e.model.layers*14  # Metadata-only CPU mock, no model execution.
        prefixes=[100+i for i in range(16)];owners=[];positions=[]
        for owner,p in enumerate(prefixes):
            values=[p-1,p] if owner<10 else [p]
            owners.extend([owner]*len(values));positions.extend(values)
        pos,meta=e._prepare_attention_metadata(owners,positions,False)
        x,pos,meta=NativeACLGraphRunner._pad_inputs(torch.zeros(26,dtype=torch.long),pos,meta,28)
        protected,index=witness.query_context(e,[pos],[meta])
        self.assertEqual(protected,[p-int(i<10) for i,p in enumerate(prefixes)])
        self.assertEqual(len(index),26);self.assertTrue(bool((index>=0).all()))
        bad=copy.copy(meta);bad.slot_mapping=meta.slot_mapping.clone();bad.slot_mapping[-1]=-2
        with self.assertRaisesRegex(RuntimeError,'PAD'):witness.query_context(e,[pos],[bad])
        tail_pos=[];tail_meta=[]
        for step in range(3):
            pp,mm=e._prepare_attention_metadata(list(range(16)),[p+step+1 for p in prefixes],False)
            tail_pos.append(pp);tail_meta.append(mm)
        protected,index=witness.query_context(e,tail_pos,tail_meta)
        self.assertEqual(protected,[p+1 for p in prefixes]);self.assertEqual(len(index),48)

    def test_scoped_witness_requires_one_true_changed_replay_and_no_fallback(self):
        import torch
        import four_family_native_witness as witness
        before=dict(exists=True,object_id=17,runtime_validated=False,validated_real_row_count=None)
        after=dict(before,runtime_validated=True)
        first=dict.fromkeys(witness.REQUIRED_COUNTERS,0);second=dict(first,runtime_validation_calls=1,changed_input_validation_calls=1)
        self.assertEqual(witness.validate_transition(before,after,first,second,kind='target',real_rows=16)['runtime_validation_calls'],1)
        self.assertIsNone(after['validated_real_row_count'])
        for field in ('eager_fallback_calls','disabled_entry_calls','unclassified_calls','runtime_validation_failures'):
            bad=dict(second);bad[field]=1
            with self.assertRaises(RuntimeError):witness.validate_transition(before,after,first,bad,kind='target',real_rows=16)
        with self.assertRaises(RuntimeError):witness.validate_transition(before,dict(after,object_id=18),first,second,kind='target',real_rows=16)
        with self.assertRaises(RuntimeError):witness.validate_transition(before,after,first,second,kind='draft',real_rows=16)
        bad=dict(first);bad.pop('eager_fallback_calls')
        with self.assertRaises(RuntimeError):witness.counters(NS(execution_counters={'draft':bad}),'draft')
        bad=dict(first);bad['runtime_validation_calls']=False
        with self.assertRaises(RuntimeError):witness.counters(NS(execution_counters={'draft':bad}),'draft')
        # A closed entry is observed without replacing the original return by
        # the diagnostic clone or fabricating a hidden real-row validation.
        key=('target-hidden:0|steps:1|paged',(24,));entry=NS(runtime_validated=True)
        runner=NS(target_entries={key:entry},execution_counters={'target':dict(first)},
            _frontier_qualification_last_key=key,last_target_execution=NS(mode='replay',used_aclgraph=True))
        engine=NS(graph_runner=runner,rank=0);reports=[]
        observer=witness.ObserveAndQualify(engine,write=lambda name,row:reports.append(row),label='CPU-only')
        original=torch.ones(24,2);saved=original.clone();meta=NS(actual_seq_lengths_q=(24,),slot_mapping=torch.arange(24))
        def no_alternate(*args):raise AssertionError('Closed entry must not run synthetic work')
        returned=observer._close('target',key,torch.zeros(24,dtype=torch.long),[torch.arange(24)],
            [meta],4,{},saved,no_alternate,original_return=original)
        self.assertIs(returned,original);self.assertIsNot(returned,saved)
        self.assertEqual((reports[-1]['owner_count'],reports[-1]['real_rows']),(16,24))
        self.assertIsNone(reports[-1]['state_before']['validated_real_row_count'])
        runner.last_target_execution=NS(mode='eager',used_aclgraph=False)
        with self.assertRaisesRegex(RuntimeError,'authoritative Native graph'):
            observer._close('target',key,torch.zeros(24,dtype=torch.long),[torch.arange(24)],
                [meta],4,{},saved,no_alternate,original_return=original)

    def test_native_protocol_preserves_all_per_request_grants_and_original_guards(self):
        import four_family_protocol as protocol,four_family_scope as scope,native_protocol
        from vllm_ascend.spec_decode.pearl import native_engine as native
        case=scope.cases()[0];e,padded,_=prepared(case)
        contexts=[dict(token_ids=[1]*129,slo_class='normal',output_offset=96) for _ in range(16)]
        source=dict(ids=list(range(16)),contexts=contexts)
        state=protocol.setup(native,source,padded[0][:16],padded[2][:16],[.5]*16,case['budgets'])
        self.assertEqual([t.gamma for t in state['tickets']],list(case['budgets']))
        self.assertEqual(len(state['payloads']),16)
        compiled=native_protocol.compile_protocol(native)
        self.assertTrue(compiled['extracted_structural_and_numerical_lanes_unchanged'])
        self.assertEqual(len(compiled['original_decode_sha256']),64)


if __name__=='__main__':unittest.main()
