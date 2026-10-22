"""HOST negative controls for the external auditor; no device/workload proof."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import analyze_v96_actual_capacity64 as audit


class ActualAuditHost(unittest.TestCase):
    def test_artifact_SHA_and_path_are_hard_boundaries(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            path=Path(directory)/'actual.json.gz'
            raw=b'{"actual":true}'
            path.write_bytes(gzip.compress(raw,mtime=0))
            item=audit.descriptor(path)|{'uncompressed_bytes':len(raw)}
            self.assertEqual(audit.read_gzip(item,directory),{'actual':True})
            with self.assertRaisesRegex(audit.Rejected,'SHA'):
                audit.read_gzip(item|{'sha256':'0'*64},directory)
            with self.assertRaisesRegex(audit.Rejected,'escaped'):
                audit.read_gzip(item,outside)
            with self.assertRaisesRegex(audit.Rejected,'length'):
                audit.read_gzip(item|{'uncompressed_bytes':len(raw)+1},directory)

    def test_percentile_method_and_empty_population(self):
        self.assertEqual(audit.percentile_linear([0,10],.9),9.)
        self.assertEqual(audit.percentile_linear([4],.9),4.)
        with self.assertRaises(audit.Rejected):audit.percentile_linear([],.9)
        with self.assertRaises(audit.Rejected):audit.percentile_linear([float('nan')],.9)

    def test_absent_raw_clocks_TPOT_pair_attainment_and_stream_are_checked(self):
        requests=[dict(request_id='actual0',slo_class='tight',slo_tpot_ms=40.),
                  dict(request_id='actual1',slo_class='normal',slo_tpot_ms=50.)]
        results=[]
        for request,paper in zip(requests,(40.,51.)):
            results.append(request|dict(completion_token_ids=[1]*256,paper_tpot_ms=paper,
                observed_tpot_ms=paper*256/255,paper_slo_attained=paper<=request['slo_tpot_ms'],
                paper_slo_goodput_tokens=256 if paper<=request['slo_tpot_ms'] else 0))
        events=[]
        for owner,elapsed,values,finished in ((0,1.,[1],False),(1,2.,[1],False),
                                            (0,10.,[1]*255,True),(1,15.,[1]*255,True)):
            events.append(dict(request_index=owner,request_id=requests[owner]['request_id'],
                               elapsed_seconds=elapsed,token_ids=values,finished=finished))
        trace=dict(generate_results=results,elapsed_seconds=20.,host_generate_call_seconds=20.,stream_events=events)
        value=audit.metric_rows(trace,requests)
        self.assertEqual(value['paper_goodput_tokens_per_second'],12.8)
        self.assertEqual(value['raw_tokens_per_second'],25.6)
        self.assertEqual(value['paper_TPOT_ms_median'],45.5)
        self.assertAlmostEqual(value['paper_TPOT_ms_P90'],49.9)
        self.assertFalse(value['raw_finish_start_clocks_available'])
        self.assertFalse(value['independent_raw_clock_reconstruction_qualified'])
        bad=copy.deepcopy(trace);bad['generate_results'][1]['paper_slo_attained']=True
        with self.assertRaisesRegex(audit.Rejected,'attainment'):audit.metric_rows(bad,requests)
        bad=copy.deepcopy(trace);bad['generate_results'][0]['paper_tpot_ms']+=1.
        with self.assertRaisesRegex(audit.Rejected,'denominator'):audit.metric_rows(bad,requests)
        bad=copy.deepcopy(trace);bad['stream_events'][-1]['token_ids'].pop()
        with self.assertRaisesRegex(audit.Rejected,'complete256'):audit.metric_rows(bad,requests)
        with self.assertRaisesRegex(audit.Rejected,'timer'):
            audit.metric_rows(trace|{'elapsed_seconds':19.},requests)

    def test_actual_production_AST_clock_fields_are_not_invented(self):
        proof=audit.production_tpot_contract()
        self.assertTrue(proof['passed'])
        self.assertFalse(proof['raw_finish_start_clocks_available_in_results'])

    def test_fusion_accepts_repeated_actual_graph_sessions_and_rejects_unknown(self):
        record='{"graph_fusion":{"TransdataCastFusionPass":{"effect_times":"0","match_times":2}}}'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fusion_result.json'
            path.write_text('{"session_and_graph_id_1_2":'+record+',"session_and_graph_id_1_2":'+record+'}')
            self.assertEqual(audit.fusion_descriptor(path)['graph_records'],2)
            path.write_text('{"session_and_graph_id_1_2":{"graph_fusion":{"ArbitraryPass":{"effect_times":0,"match_times":0}}}}')
            with self.assertRaisesRegex(audit.Rejected,'unexpected'):audit.fusion_descriptor(path)

    def test_missing_all64_KV_proof_rejects_numerical_summary(self):
        phase=dict(geometry=dict(N=16,physical_M=197,physical_S=19,mask_shape=[19,1,128,4096],
            prompt_start=2099,prompt_end=2196,final=False),phase_index=0,actual_committed_cursor=2196,
            publication=dict(published=False),target_numeric={k:True for k in ('passed',
            'all197_hidden_original_matcher','all64_target_query_KV_original_matcher',
            'exact_tree_and_final_prompt_heads','all_candidate_and_reference_prefixes_finite_and_unchanged',
            'actual_query_KV_restored','reference_never_initialized_from_candidate_KV')})
        phase['target_numeric']['independent_eager_calls']=[{}]*19
        self.assertEqual(audit.phase_summary(phase,1)['all_layer_KV_matcher_observed'],64)
        del phase['target_numeric']['all64_target_query_KV_original_matcher']
        with self.assertRaisesRegex(audit.Rejected,'all64'):audit.phase_summary(phase,1)

    def test_frozen_report_validators_import_without_torch_or_Native(self):
        with audit.frozen_host_modules() as modules:
            self.assertTrue(callable(modules['bounded_report_audit_v78'].chain_record))
            self.assertTrue(callable(modules['audit_capacity64_v96'].audit_worker_reports))
            with self.assertRaisesRegex(audit.Rejected,'forbids'):
                __import__('torch_npu')

    def test_running_receipt_is_not_a_terminal_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'local-receipt-running.json'
            path.write_text(json.dumps(dict(schema='V96.capacity64-runtime-supervisor.v1',status='running')))
            with self.assertRaisesRegex(audit.Rejected,'terminal'):audit.analyze('running',Path(directory))

    def test_actual_capacity_and_page_accounting_reject_reported_shortcuts(self):
        trace=dict(active_peak=33,actual_capacity64_scope_completed=True,
            allocator=dict(pool_blocks=544,referenced_page_highwater=500,final_referenced_pages=0,
                acquired_pages=700,released_pages=700,allocation_storage_rows=[527]),
            decode_counters=dict(spec_rhythm_peak_active_requests=33),native={},phase_seconds={},
            normal_target_calls=[dict(N=17,physical_M=116,physical_S=24)],standalone_prefill_calls=[],
            logical_events=[],source_generation=4,source_generation_before=3,
            sealed_before=True,sealed_after=True,cache_released=True)
        self.assertEqual(audit.trace_observation(trace)['normal_target_M_S_distribution'],{'116/24':1})
        bad=copy.deepcopy(trace);bad['active_peak']=32;bad['decode_counters']['spec_rhythm_peak_active_requests']=32
        with self.assertRaisesRegex(audit.Rejected,'Configured64'):audit.trace_observation(bad)
        bad=copy.deepcopy(trace);bad['allocator']['released_pages']-=1
        with self.assertRaisesRegex(audit.Rejected,'page/storage'):audit.trace_observation(bad)
        bad=copy.deepcopy(trace);bad['source_generation']=5
        with self.assertRaisesRegex(audit.Rejected,'epoch'):audit.trace_observation(bad)


if __name__=='__main__':
    unittest.main()
