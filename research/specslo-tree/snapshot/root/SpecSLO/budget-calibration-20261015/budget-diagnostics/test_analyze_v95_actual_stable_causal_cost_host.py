"""Small parser-only regressions. Historical rows are HOST fixtures, not V95 evidence."""
import copy
import importlib.util
import json
import itertools
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v95_external_parser',P/'analyze_v95_actual_stable_causal_cost.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
OLD = a.BASE/'tree-four-family-budget-cost-transport-v76/context-cost-sweep-v1/four-family-budget-cost-transport-v1'


class ParserHOST(unittest.TestCase):
    def _actual_route_row(self,method='stable_staged80'):
        return dict(method=method,actual_target_route=dict(requested_arm=method,
            executed='generic_staged',native_execution_family='generic',native_execution_mode='replay',
            generic_entry_key=a.expected_keys(method)['greedy'],source_native_forward_sha256='a'*64,
            same_M_independent_qualification_inherited=False,production_overlap_qualified=False,formal_Goodput=False),
            original_native_execution_deltas={family:dict.fromkeys(a.EXECUTION_FIELDS,0)
                for family in ('generic','draft','target')})

    def test_complete_raw_MC2_schedule_has57_records10_independent_refs_and15_cost(self):
        schedule=a.raw_schedule()
        self.assertEqual(len(schedule),57)
        self.assertEqual(sum(label=='independent-eager' for name,label in schedule),10)
        self.assertEqual(schedule[33:36],[('stable_staged80',label) for label in ('candidate','independent-eager','plain-target-replay')])
        self.assertEqual(len(schedule[-15:]),15)

    def test_generic_entry_key_remains_distinct_from_target_and_serial_at_sameM80(self):
        generic=a.expected_keys('stable_materialized80')
        target=a.expected_keys('target80')
        self.assertNotEqual(generic['greedy'],target['greedy'])
        self.assertIn('width:5|verify:16',generic['greedy'])
        self.assertNotIn('width:4|verify:20',generic['greedy'])
        self.assertTrue(generic['hidden'].startswith("('hidden|fia-stable:"))

    def test_real_generic_counter_fields_accept_one_generic_replay_but_reject_target_relabel(self):
        row=self._actual_route_row()
        row['original_native_execution_deltas']['generic'].update(total_calls=1,replay_calls=1)
        a.actual_execution_counters(row,1,sealed=True,native_forward_sha='a'*64)
        row['original_native_execution_deltas']['target']=row['original_native_execution_deltas']['generic']
        row['original_native_execution_deltas']['generic']=dict.fromkeys(a.EXECUTION_FIELDS,0)
        with self.assertRaisesRegex(RuntimeError,'one true Native'):
            a.actual_execution_counters(row,1,sealed=True,native_forward_sha='a'*64)

    def test_missing_generic_native_counter_or_wrong_staging_phase_is_rejected(self):
        row=self._actual_route_row()
        row['original_native_execution_deltas']['generic'].update(total_calls=1,replay_calls=1)
        row['original_native_execution_deltas']['generic'].pop('changed_input_validation_calls')
        with self.assertRaisesRegex(RuntimeError,'all ten'):
            a.actual_execution_counters(row,1,sealed=True,native_forward_sha='a'*64)
        with self.assertRaisesRegex(RuntimeError,'actual target versus'):
            a.route_proof(self._actual_route_row(),1,sealed=False,native_forward_sha='a'*64)

    def test_raw_generic_MC2_keys_cannot_alias_old_serial_key(self):
        keys=a.expected_keys('stable_staged80')
        raw=dict(history=dict(case='stable_staged80',keys=keys),actual_greedy_key=keys['greedy'],
            hidden_executed_in_this_interval=False,actual_hidden_key='None',after_entries={
                kind:dict(physical_M=80,sequence_S=16,tree=False,native_family='generic',
                    checks=dict(all80rows=True,persistent_staging=True)) for kind in ('greedy','hidden')})
        a.mc2_entry_contract(raw,'stable_staged80')
        raw['actual_greedy_key']=raw['actual_greedy_key'].replace('width:5|verify:16','width:4|verify:20')
        with self.assertRaisesRegex(RuntimeError,'actual observed greedy'):
            a.mc2_entry_contract(raw,'stable_staged80')

    def test_summary_preserves15_synchronized_calls_and_each_actual_g_without_rank_pooling(self):
        reports=[]
        for rank in range(4):
            rows=[]
            for sample in range(5):
                rotated=a.NAMES[sample%3:]+a.NAMES[:sample%3]
                for name in (tuple(reversed(rotated)) if sample%2 else rotated):
                    rows.append(dict(method=name,sample=sample,case=dict(name=name,budgets=[4]*16,C=64,Q=80,M=80,S=16,family='causal',scratch_Q=[]),
                        g=[1]*16,accepted=[[-1]*4 for _ in range(16)],emitted=[[8,-1,-1,-1,-1] for _ in range(16)],
                        selected_original_nodes=[list(range(4)) for _ in range(16)],plain_path_signature={'HOST_fixture':True},
                        local_phase_wall_ms=dict.fromkeys(a.PHASES,1.+rank),
                        sequential_local_round_plus_nextproposal_wall_ms=10.+rank,
                        next_missing_suffix_lengths=[0]*16,next_draft_query_count=64,next_draft_counters={}))
            by_name={name:next(r for r in rows if r['method']==name) for name in a.NAMES}
            cross=[a.signature_differences(x,y,by_name[x],by_name[y]) for x,y in itertools.combinations(a.NAMES,2)]
            reports.append(dict(cost=rows,cross_layout_actual_emission_differences=cross))
        value=a.summarize(reports)
        self.assertEqual(value['synchronized_cost_calls'],15)
        self.assertTrue(value['rank_observations_are_not_independent_samples'])
        self.assertEqual([r['sum_actual_delivered_tokens'] for r in value['family_summary']],[16]*3)
        self.assertEqual(value['family_summary'][0]['rank1_target_cost']['median_ms'],2.)

    def test_running_or_failed_receipt_refuses_before_source_or_result_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            w = Path(temporary)
            for status in ('running','failed'):
                (w/'cost-receipt-stable-causal-three-arm-cost-test.json').write_text(json.dumps(
                    dict(schema='v95-stable-causal-cost-receipt-v1',status=status)))
                with patch.object(a,'WORK',w),patch.object(a,'sha',side_effect=AssertionError('too early')):
                    with self.assertRaisesRegex(RuntimeError,'terminal complete'):
                        a.audit('stable-causal-three-arm-cost-test')
                self.assertEqual(sorted(p.name for p in w.iterdir()),['cost-receipt-stable-causal-three-arm-cost-test.json'])

    def test_original_protocol_AST_hashes_include_decorator_without_importing_torch(self):
        freeze = a.read(a.WORK/a.FREEZE)
        original,canonical = a.source_protocol_hashes(freeze['external_files'])
        old = a.read(OLD/'rank0-four-family-cost.json')['qualification'][0]['Native_protocol']
        self.assertEqual((original,canonical),(old['original_decode_sha256'],old['canonical_decode_sha256']))
        self.assertNotIn('torch',sys.modules)

    def test_raw_native_witness_parser_accepts_historical_same_schema_and_rejects_false_restore(self):
        q = a.read(OLD/'rank0-four-family-cost.json')['qualification'][0]
        rows = q['original_same_key_native_witnesses']['first']
        a.native_witnesses(rows)
        bad = copy.deepcopy(rows)
        bad[-1]['restored_queryKV_native_match'] = False
        with self.assertRaisesRegex(RuntimeError,'restoration'):
            a.native_witnesses(bad)

    def test_cross_M_conflicts_remain_visible_and_are_not_equality_failures(self):
        x = dict(g=[1]*16,emitted=[[10,-1] for _ in range(16)])
        y = copy.deepcopy(x)
        y['emitted'][3] = [11,12]
        y['g'][3] = 2
        result = a.signature_differences('target80','stable_materialized80',x,y)
        self.assertEqual(result['common_valid_position_token_mismatches'][0]['owner'],3)
        self.assertEqual(result['emitted_length_differences'][0],dict(owner=3,first=1,second=2))
        self.assertEqual(result['same_grant_g_label_differences'][0]['second_g'],2)
        self.assertFalse(result['pooling_into_gfit_qualified'])

    def test_missing_and_nonfinite_times_are_not_filled_in(self):
        for values in ([],[float('nan')],[float('inf')],[-1],[True]):
            with self.assertRaises(RuntimeError):
                a.stats(values)
        self.assertEqual(a.stats([3.,1.,2.])['median_ms'],2.)

    def test_known_fusion_output_keeps_duplicate_keys_and_does_not_claim_writer(self):
        passes = {name:dict(effect_times='0',match_times='1') for name in
            ('TransdataCastFusionPass','TransdataFz2FzgFusionPass','TransdataFzg2FzFusionPass')}
        record = json.dumps(dict(graph_fusion=passes))
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)/'fusion_result.json'
            p.write_text('{"session_and_graph_id_0_0":'+record+',"session_and_graph_id_0_0":'+record+'}')
            result = a.postrun_top_level(temporary,set(),started_unix=0,completed_unix=time.time()+1)
            self.assertEqual(len(result),1)
            self.assertEqual((result[0]['object_pair_records'],result[0]['unique_session_graph_keys']),(2,1))
            self.assertFalse(result[0]['writer_identity_qualified'])
            self.assertFalse(result[0]['consumed_source_input'])

    def test_postrun_unknown_code_or_static_JSON_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name in ('unexpected.py','unexpected.sh','unexpected.json','unexpected.so'):
                p = Path(temporary)/name
                p.write_text('{}')
                with self.assertRaisesRegex(RuntimeError,'unexpected unbound postrun'):
                    a.postrun_top_level(temporary,set(),started_unix=0,completed_unix=time.time()+1)
                p.unlink()

    def test_fusion_name_is_not_a_blanket_output_exemption(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)/'fusion_result.json'
            for malformed in ({'code':'not a fusion record'},
                    {'session_and_graph_id_0_0':{'graph_fusion':{'UnreviewedPass':{'effect_times':'0','match_times':'1'}}}}):
                p.write_text(json.dumps(malformed))
                with self.assertRaises(RuntimeError):
                    a.postrun_top_level(temporary,set(),started_unix=0,completed_unix=time.time()+1)

    @classmethod
    def tearDownClass(cls):
        if 'torch' in sys.modules:
            raise AssertionError('HOST parser regressions imported torch')


if __name__ == '__main__':
    unittest.main()
