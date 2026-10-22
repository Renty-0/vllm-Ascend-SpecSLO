"""Pure HOST parser controls. Synthetic fixtures never become device evidence."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('V102_external_parser_HOST',P/'analyze_v102_actual_packed_eager01_cost.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def static():
    return dict(passed=True,checks={k:True for k in ('disk_raw_sha','normalized_manifest_canonical_sha','profile_operator',
        'standalone_chained_flush','all64layers','route_count128','routes_frozen','all3physical_rows_in_chained_flush_rows',
        'every_route_enabled_and_all3physical_rows_qualified')},bad_routes=[],profile_raw_sha256=a.PROFILE_SHA,
        profile_canonical_sha256='67feca4c924ce95b252812d2db606a65a43942c472c429a71d12e5270442deb7',
        manifest_canonical_sha256='67feca4c924ce95b252812d2db606a65a43942c472c429a71d12e5270442deb7',
        chained_flush_rows=a.MC2_ROWS,deferred_rows_diagnostic_only=[],deferred_read_done_chain_required=False)


def mc2(g=None):
    values=dict.fromkeys(a.MC2_FIELDS,0);values.update(native_epilogue_success=128)
    r=dict(passed=True,static_contract=static(),dispatch=dict(passed=True,native_success_required=True,
        before=dict.fromkeys(a.MC2_FIELDS,0),after=values,values=values))
    if g is not None:
        keys=a.target_keys(g);before={k:dict(exists=False,valid=False,object_id=None,checks={}) for k in keys}
        after={k:dict(exists=True,valid=True,object_id=i+1,checks=dict(HOST_only=True)) for i,k in enumerate(keys)}
        r.update(history=dict(geometry=g,vocabulary=a.VOCAB,keys=keys,history_valid=True,entries=before,cold_capture_required=True),
            checks=dict(HOST_only=True),after_entries=after,actual_greedy_key=keys['greedy'],hidden_key=keys['hidden'],
            hidden_executed_in_interval=True,hidden_key_evidence='actual_qualification_execution')
    return r


def target(child=False):
    return dict({k:True for k in ('original_allhidden_allKV_1e3','exact_tokens','native_changed_input_gate','all_owner_prefixes_protected')},
        geometry=a.target_geometry(child),MC2=dict(candidate=mc2(a.target_geometry(child)),fresh_eager_reference=mc2()))


def compact():return dict(all_real_rows=True,all_layers=True,independent_source_KV=True,perrow_original_query_offset=True,dense_AR_qualified=False)


def signature(c,method,changed=False):
    n=len(c['normal_repairs']);rows=[[11,12,13,14,21,22,23,24] for _ in range(n+1)]
    scores=[[a.float32(v) for v in [.6]*4+[.2]*4] for _ in rows]
    if method=='packed_seed_tail3':
        for j in (3,7):rows[-1][j]=0;scores[-1][j]=0
    contexts=a.source_contexts(c)
    roots=[(int(contexts[i]['token_ids'][c['prefix']+r])+int(changed))%a.VOCAB for i,r in enumerate(c['normal_repairs'])]+[31]
    return dict(draft_token_ids=rows,root_token_ids=roots,eager_frontier_tokens={str(n):31},node_confidences=scores,draft_confidence=[a.float32(.4)]*len(rows))


def ledger(c,method,out,changed):
    P=c['prefix'];n=len(c['normal_repairs']);v={}
    for i,r in enumerate(c['normal_repairs']):
        p=P+r;v[str(i)]=dict(safe_before=p,tokens={str(p+j):t for j,t in enumerate([out['root_token_ids'][i],*out['draft_token_ids'][i][:3]])})
    values=[(int(a.source_contexts(c)[n]['token_ids'][P])+int(changed))%a.VOCAB,101,102,103,104,out['root_token_ids'][n],*out['draft_token_ids'][n][:2 if method=='packed_seed_tail3' else 3]]
    v[str(n)]=dict(safe_before=P,tokens={str(P+j):t for j,t in enumerate(values)})
    return v


def row(rank,c,method,changed=False,timed=False):
    r=dict(case=c,method=method,changed=changed,actual_call=(rank==0),target_or_whole_method_latency_ms=None,actual_delivered_g=None,
        draft_latency_ms=None,source_identity_sha256=None,signature=None)
    if rank:return r
    z=dict.fromkeys(a.GATES,0);g=a.validate_case(c);out=signature(c,method,changed)
    before={k:dict.fromkeys(a.EXECUTION_FIELDS,0) for k in ('generic','target','draft')};after=copy.deepcopy(before)
    after['draft'].update(total_calls=1,replay_calls=1)
    after['target'].update(total_calls=1 if method=='packed_seed_tail3' else 2,replay_calls=1 if method=='packed_seed_tail3' else 2)
    identity=hashlib.sha256(json.dumps([c,changed],sort_keys=True).encode()).hexdigest()
    q=g['first_M']+3*g['head_M'] if method=='packed_seed_tail3' else a.bucket(sum(c['normal_repairs']),64)+1+4*g['head_M']
    r.update(signature=out,final_known_ledger=ledger(c,method,out,changed),source_identity_sha256=identity,
        initial_source_prefix_hashes=['b'*64]*56,Native_before=z,Native_after_fixture=z,Native_after=z,Native_deltas=z,Native_timed_call_deltas=z,
        execution_before=before,execution_after=after,actual_method_query_count=q,actual_method_graph_calls=2 if method=='packed_seed_tail3' else 3,
        actual_method_model_calls=2 if method=='packed_seed_tail3' else 3,draft_latency_ms=(5.0 if method=='original_fast4' else 4.0) if timed else None,
        untimed_replay_elapsed_ms=None if timed else 4.,sealed=True,
        measured_method_includes='actual draft function: metadata, model graphs, outside head if any, original D2H and result packing; final NPU synchronize',
        excluded_from_timer='source cache release/allocate/prefill, actual B5 eager parent, independent references, source/proof checks, rank coordination')
    return r


def nextproof():
    known=dict(runtime_validated=True,validated_real_row_count=14)
    warm=[dict(delta=v,roots=[7+v]*14,key='HOST-only-key',root_mode=True,head_rows=16,
        source_history_physically_replayed=True,runtime_validated=True,validated_real_row_count=14,
        qualification_before_three_calls=known,actual_seed_tokens=[v]*14) for v in (0,1,0)]
    return dict({k:True for k in a.FUTURE_FLAGS},prefixes=[133]*14,starts=[132]*14,missing_suffix_lengths=[1]*14,
        dense_new_prefix_prefill_used=False,candidate_reference_known_prefix_cross_bitwise_equality_claimed=False,native_warmup=warm)


def protocol(c,outcome,rank):
    rows=[[7,-1,-1,-1,-1] for _ in range(14)];accepted=[[-1]*4 for _ in rows]
    q=dict(case=c,prefix=128,normal=13,eager=1,outcome=outcome,qualification_inherited=False,
        Native_original_finite_preflight_commit_promotion_guards=True,actual_method='V102-packedPA-normalB5-eagerB3-ragged01-tail3',
        production_dual_home_overlap=False,dense_canonical_AR_qualified=False,cross_M_equality_qualified=False,
        acceptance_rate_gain_or_Goodput_measured=False,online_qualified=False,
        proposal_exchange=dict(original_WORLD_exchange=True,payload_ticket_identity_retained=True,parent_rows=14,staged_child_rows=1,ordered_proposal_ids=list(range(15))),
        actual_observed_parent_output=copy.deepcopy(rows),actual_observed_parent_accepted=copy.deepcopy(accepted),
        exercised_parent_output=copy.deepcopy(rows),exercised_parent_accepted=copy.deepcopy(accepted),
        parent_counterfactual_protocol_only=(outcome!='observed_target'),promoted_owners=[],real_promoted_child_consume=None,
        actual_nextrepair=nextproof() if rank==0 else None,parent_target=target() if rank else None,parent_compaction=compact() if rank else None)
    if outcome=='valid_promotion':
        q['promoted_owners']=[13];q['exercised_parent_output'][13]=[11,12,13,14,31];q['exercised_parent_accepted'][13]=[0,1,2,3]
        per=[]
        for i,path in enumerate(q['exercised_parent_accepted']):
            nodes=[v for v in path if v>=0];root=128+128*i
            per.append(dict(owner=i,selected_nodes=nodes,source_slots=[root,*[root+v+1 for v in nodes]],destination_slots=list(range(root,root+len(nodes)+1))))
        history=[dict(mode='fresh-independent-parent-and-own-index-copy',MC2=mc2(),per_owner=per,
            fresh_independent_source_only=True,clone_before_index_copy=True,candidate_KV_never_used_as_reference=True),
            dict(mode='original-Native-parent-and-Native-compaction',MC2=mc2(a.target_geometry()),reference_KV_copied_back_to_candidate=False)]
        q['real_promoted_child_consume']=dict(actual_observed_only=True,normal_owner_prefixes_protected=True,consumed_owners=[13],
            observed_output=[[8,-1,-1,-1,-1]],observed_accepted=[[-1]*4],target_proof=target(True) if rank else None,
            committed_KV=compact() if rank else None,independent_parent_history=history if rank else [])
    if outcome=='same_token_changed_ancestor':q['exercised_parent_output'][13]=[21,31,-1,-1,-1];q['exercised_parent_accepted'][13]=[4,-1,-1,-1]
    return q


def report(rank):
    z=dict.fromkeys(a.GATES,0);r=dict(schema='V102.ragged01-packed-qualification-and-local-draft-cost.v1',rank=rank,errors=[],gain=[],
        sealed=True,graph_sealed=True,NPU_qualified_bounded_ragged01_samples_only=True,timing_qualified_local_draft_only=True,
        actual_delivered_g=None,Goodput=None,production_dual_home_overlap=False,online_qualified=False,dense_AR_qualified=False,
        expected_packed_numerical=18,expected_control_numerical=9,expected_protocol=12,expected_synchronized_cost_experiments=30,
        preseal_Native=z,fixed_point=[],timing=[],protocol=[protocol(c,o,rank) for c in a.protocol_cases() for o in a.OUTCOMES])
    r['warmup']=[dict(case=c,method=m,calls=[signature(c,m,ch) if rank==0 else None for ch in (False,True,False)]) for c,m in [(c,'packed_seed_tail3') for c in a.cases()]+[(c,'original_fast4') for c in a.cost_cases()]]
    for field,cs,method in (('numerical',a.cases(),'packed_seed_tail3'),('control_numerical',a.cost_cases(),'original_fast4')):
        r[field]=[]
        for c in cs:
            for ch in (False,True,False):
                g=a.validate_case(c);q=dict({k:True for k in a.NUMERIC_FLAGS},case=c,method=method,changed_input_variant=ch,changed=ch,
                    prefix=c['prefix'],normal=g['normal'],eager=1,qualification_inherited=False,timing_qualified=False,
                    native_keys=[dict(kind=k,key=v,runtime_validated=True,validated_real_row_count=g['total'] if k=='draft' else None) for k,v in a.expected_native_keys(c,method)])
                if method=='packed_seed_tail3':q.update(original_before_tail_real_first=True,root_head_outside_graph=True,
                    diagnostic_observer_scope_only=True,immediate_first_hidden_observer_numeric_only=True,first_M=g['first_M'],tail_M=g['head_M'],
                    eager_uncomputed_indices=[3,7],eager_aggregate_confidence_denominator=6,normal_aggregate_confidence_denominator=8)
                else:q.update(frontier_capture_eager_storage_nonalias=True,diagnostic_observer_scope_only=True,actual_method='V46-original-FastV5-repairPA-frontierPA-chain4')
                r[field].append(q)
    for sweep in range(2):
        points=[(c,m) for c in a.cost_cases() for m in a.METHODS]
        if sweep:points.reverse()
        r['fixed_point'].append(dict(sweep=sweep,Native_before=z,Native_after=z,Native_deltas=z,rows=[row(rank,c,m,ch) for c,m in points for ch in (False,True,False)]))
    for repeat in range(5):
        for c in a.cost_cases():
            for m in (a.METHODS if repeat%2==0 else tuple(reversed(a.METHODS))):
                v=row(rank,c,m,timed=True);v.update(repeat=repeat);r['timing'].append(v)
    r['startup_memory']=r['final_memory']=dict(layers=28 if rank==0 else 64,blocks=544,total=64*1024**3,free=8*1024**3,reserved=40*1024**3,peak=40*1024**3,entries=6)
    r['final_entry_states']=[dict(runtime_validated=True)];r['final_graph_metrics']=z
    return r


class ExternalHOST(unittest.TestCase):
    def test_complete_synthetic_HOST_reports_pass_independent_deep_parser(self):
        reports=[report(i) for i in range(4)];c=dict(replies=[['calibrated',rank,dict(rows=0,timings=30)] for rank in range(4)])
        self.assertEqual(a.reports_deep(reports,c),[])

    def test_real_historical_same_schema_MC2_and_source_hashes_parse_without_torch(self):
        base=a.BASE/'tree-packed-root-seed-eager3-v65/context-cost-sweep-v1/packed-eager3-qualification-v1'
        old=a.read(base/'rank1-packed-eager3-qualification.json')['protocol'][0]['parent_target']
        a.raw_mc2(old['MC2']['candidate'],geometry=old['geometry']);a.raw_mc2(old['MC2']['fresh_eager_reference'],independent=True)
        actual=a.read(base/'rank1-packed-eager3-qualification.json')['protocol']
        for q in actual:
            a.verdict(q['actual_observed_parent_output'],q['actual_observed_parent_accepted'],q['normal']+q['eager'])
            if q['real_promoted_child_consume']:
                child=q['real_promoted_child_consume'];a.verdict(child['observed_output'],child['observed_accepted'],len(child['consumed_owners']),child=True)
        value=a.source_protocol_hashes(a.read(a.WORK/a.FREEZE)['external_files']);self.assertTrue(a.hexsha(value['original_decode_sha256']))
        self.assertFalse(value['per_protocol_record_function_hashes_persisted']);self.assertNotIn('torch',sys.modules)

    def test_running_and_failed_receipts_refuse_before_source_result_access(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for status in ('running','failed'):
                (p/'cost-receipt-packed-eager01-HOST.json').write_text(json.dumps(dict(schema='V102.ragged01-packed-cost-receipt.v1',status=status)))
                with patch.object(a,'WORK',p),patch.object(a,'source_and_HOST',side_effect=AssertionError('too early')):
                    with self.assertRaisesRegex(RuntimeError,'terminal complete'):a.audit('packed-eager01-HOST')
                self.assertEqual(len(list(p.iterdir())),1)

    def test_three_costs_preserve15pairs30rank0calls_and_null_targetcost(self):
        reports=[report(i) for i in range(4)];s=a.summarize(reports)
        self.assertEqual((s['paired_comparisons'],s['actual_rank0_draft_method_calls']),(15,30))
        self.assertEqual(s['cases'][0]['draft_whole_method_ms']['packed_seed_tail3']['median_ms'],4.)
        self.assertIsNone(s['cases'][0]['target_latency_ms']);self.assertIsNone(s['actual_delivered_g'])
        self.assertFalse(s['cases'][-1]['service_C96_valid']);self.assertEqual(s['cases'][-1]['metadata_scope']['draft_candidates'],163)

    def test_numeric_missing_KV_or_native_realrow_gate_cannot_inherit_old_scope(self):
        for kind in ('KV','realrow','M','schedule'):
            r=report(0)
            if kind=='KV':r['numerical'][0]['all_layer_query_KV']=False
            if kind=='realrow':r['numerical'][0]['native_keys'][-1]['validated_real_row_count']=13
            if kind=='M':r['numerical'][0]['first_M']=32
            if kind=='schedule':r['numerical'].pop()
            with self.assertRaises(RuntimeError):a.numeric_records(r)

    def test_original_and_packed_native_keys_and_physical_costs_are_distinct(self):
        self.assertEqual([a.validate_case(c)['first_M'] for c in a.cost_cases()],[20,24,64])
        self.assertEqual([a.validate_case(c)['head_M'] for c in a.cost_cases()],[16,16,36])
        for c in a.cost_cases():
            self.assertNotEqual(a.expected_native_keys(c,'original_fast4'),a.expected_native_keys(c,'packed_seed_tail3'))
            for m in a.METHODS:a.plain_row(row(0,c,m,timed=True),0,timed=True)

    def test_target_nullable_cost_and_no_fake_rank_trial_are_required(self):
        c=a.cost_cases()[0]
        for field,value in (('draft_latency_ms',1.),('signature',{}),('source_identity_sha256','a'*64),('actual_call',True),('actual_delivered_g',3)):
            r=row(1,c,'original_fast4',timed=True);r[field]=value
            with self.assertRaises(RuntimeError):a.plain_row(r,1,timed=True)

    def test_generic_counter_relabel_and_original_extra_validation_are_rejected(self):
        c=a.cost_cases()[0]
        for field in ('generic','validation','missing'):
            r=row(0,c,'packed_seed_tail3',timed=True)
            if field=='generic':r['execution_after']['generic'].update(total_calls=1,replay_calls=1)
            if field=='validation':r['execution_after']['target']['runtime_validation_calls']=1
            if field=='missing':r['execution_after']['target'].pop('unclassified_calls')
            with self.assertRaises(RuntimeError):a.execution_delta(r,'packed_seed_tail3')

    def test_nonzero_discovery_and_missing_21st_gate_are_rejected(self):
        z=dict.fromkeys(a.GATES,0)
        for bad in (z|{'aclgraph_captures':1},{k:v for k,v in z.items() if k!='aclgraph_generic_runtime_validation_failures'}):
            with self.assertRaises(RuntimeError):a.native_delta(z,bad,zero=True)

    def test_bad_eager_ledger_or_unused_slot_or_confidence_denominator_fails(self):
        c=a.cost_cases()[0]
        for kind in ('ledger','slot','confidence','root'):
            r=row(0,c,'packed_seed_tail3',timed=True);n=len(c['normal_repairs'])
            if kind=='ledger':r['final_known_ledger'][str(n)]['tokens'][str(c['prefix']+8)]=9
            if kind=='slot':r['signature']['draft_token_ids'][-1][3]=9
            if kind=='confidence':r['signature']['draft_confidence'][-1]=a.float32(.3)
            if kind=='root':r['signature']['root_token_ids'][0]+=1
            with self.assertRaises(RuntimeError):a.plain_row(r,0,timed=True)

    def test_original_verdict_control_and_child_history_reject_false_promotion(self):
        for kind in ('changedancestor','childref','nextrestore'):
            r=report(0 if kind=='nextrestore' else 1)
            if kind=='changedancestor':r['protocol'][3]['promoted_owners']=[13]
            if kind=='childref':r['protocol'][1]['real_promoted_child_consume']['independent_parent_history'][1]['reference_KV_copied_back_to_candidate']=True
            if kind=='nextrestore':r['protocol'][0]['actual_nextrepair']['whole_method_restored']=False
            with self.assertRaises(RuntimeError):a.protocol_records(r,r['rank'])

    def test_two_complete_zero_traces_and_real_same_context_pairs_are_required(self):
        for kind in ('trace','source','prefix','oldoutput'):
            r=report(0)
            if kind=='trace':r['fixed_point'][1]['rows'].pop()
            if kind=='source':r['timing'][1]['source_identity_sha256']='a'*64
            if kind=='prefix':r['timing'][1]['initial_source_prefix_hashes'][0]='c'*64
            if kind=='oldoutput':r['timing'][0]['signature']['draft_token_ids'][0][0]=99
            with self.assertRaises(RuntimeError):a.zero_and_cost(r,0)

    def test_MC2_raw_subtraction_missing_static_and_nonpositive_reference_rejected(self):
        for kind in ('fallback','field','reference','static','key'):
            r=mc2(a.target_geometry())
            if kind=='fallback':r['dispatch']['after']['fallback']=1;r['dispatch']['values']['fallback']=1
            if kind=='field':r['dispatch']['before'].pop('exception')
            if kind=='reference':r['dispatch']['after']['native_epilogue_success']=0;r['dispatch']['values']['native_epilogue_success']=0
            if kind=='static':r['static_contract']['checks']['all64layers']=False
            if kind=='key':r['actual_greedy_key']=r['actual_greedy_key'].replace('(96,)', '(80,)')
            with self.assertRaises(RuntimeError):a.raw_mc2(r,geometry=a.target_geometry())

    def test_duplicate_known_CANN_fusion_keys_preserved_and_not_claimed_writer(self):
        old=a.BASE/'tree-stable-causal-cost-v95/fusion_result.json'
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fusion_result.json';p.write_bytes(old.read_bytes())
            r=a.fusion_output(p,started=0,completed=time.time()+1)
            self.assertEqual((r['object_records'],r['unique_session_graph_keys']),(3,1))
            self.assertFalse(r['writer_identity_inferred']);self.assertFalse(r['source_input'])

    def test_exact_frozen_collector_duplicate_regression_is_retained_without_mutating_it(self):
        source=a.WORK/'packed_snapshot_v102.py'
        spec=importlib.util.spec_from_file_location('_frozen_V102_snapshot_HOST_regression',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        before=a.sha(source)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fusion_result.json';p.write_bytes((a.BASE/'tree-stable-causal-cost-v95/fusion_result.json').read_bytes())
            with self.assertRaisesRegex(RuntimeError,'Duplicate CANN diagnostic key'):module.runtime_output(p)
            self.assertEqual(a.fusion_output(p,started=0,completed=time.time()+1)['object_records'],3)
        self.assertEqual(a.sha(source),before)

    def test_fusion_filename_does_not_exempt_unknown_structure_or_code(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fusion_result.json';p.write_text('{"code":"not fusion"}')
            with self.assertRaises(RuntimeError):a.top_level(d,set(),started=0,completed=time.time()+1)
            p.unlink()
            for name in ('extra.py','extra.json','extra.so'):
                q=Path(d)/name;q.write_text('{}')
                with self.assertRaises(RuntimeError):a.top_level(d,set(),started=0,completed=time.time()+1)
                q.unlink()

    def test_source_archive_requires_exact_flat_unique_members(self):
        import tarfile
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.tar.gz';raw=b'HOST only\n'
            with tarfile.open(p,'w:gz') as t:
                v=tarfile.TarInfo('source.py');v.size=len(raw);t.addfile(v,io.BytesIO(raw))
            self.assertEqual(a.archive_sources(p,{'source.py':hashlib.sha256(raw).hexdigest()}),1)
            with self.assertRaises(RuntimeError):a.archive_sources(p,{})

    def test_invalid_or_missing_times_are_never_filled_with_zero(self):
        for values in ([],[None],[float('nan')],[float('inf')],[-1],[True],[0]):
            with self.assertRaises(RuntimeError):a.stats(values)
        self.assertEqual(a.stats([1.,3.,2.])['median_ms'],2.)

    @classmethod
    def tearDownClass(cls):
        if 'torch' in sys.modules:raise AssertionError('HOST external audit imported torch')


if __name__=='__main__':
    stream=io.StringIO();start=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromTestCase(ExternalHOST)
    inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),P/'analyze_v102_actual_packed_eager01_cost.py')}
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    after={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in inputs}
    report=dict(schema='V102.external-actual-auditor-HOST-tests.v1',status='passed' if result.wasSuccessful() and inputs==after and 'torch' not in sys.modules else 'failed',
        actual_HOST_argv=[sys.executable,*sys.argv],tests=result.testsRun,failures=[dict(test=str(t),error=e) for t,e in result.failures],
        errors=[dict(test=str(t),error=e) for t,e in result.errors],elapsed_seconds=time.perf_counter()-start,
        input_hashes=inputs,input_hashes_after=after,actual_V102_audit_executed=False,torch_imported=('torch' in sys.modules),NPU_initialized=False)
    (P/'V102_EXTERNAL_ACTUAL_AUDITOR_HOST_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    (P/'V102_EXTERNAL_ACTUAL_AUDITOR_HOST.log').write_text(stream.getvalue())
    print(stream.getvalue(),end='');print(json.dumps({k:report[k] for k in ('status','tests','elapsed_seconds','actual_V102_audit_executed','torch_imported')},indent=2))
    raise SystemExit(report['status']!='passed')
