"""Small parser-only regressions. Historical rows are HOST fixtures, not V91 evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v91_external_parser',P/'analyze_v91_actual_causal80_cost.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
OLD = a.BASE/'tree-four-family-budget-cost-transport-v76/context-cost-sweep-v1/four-family-budget-cost-transport-v1'


class ParserHOST(unittest.TestCase):
    def test_running_or_failed_receipt_refuses_before_source_or_result_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            w = Path(temporary)
            for status in ('running','failed'):
                (w/'cost-receipt-causal80-three-case-cost-test.json').write_text(json.dumps(
                    dict(schema='v91-causal80-cost-receipt-v1',status=status)))
                with patch.object(a,'WORK',w),patch.object(a,'sha',side_effect=AssertionError('too early')):
                    with self.assertRaisesRegex(RuntimeError,'terminal complete'):
                        a.audit('causal80-three-case-cost-test')
                self.assertEqual(sorted(p.name for p in w.iterdir()),['cost-receipt-causal80-three-case-cost-test.json'])

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
        result = a.signature_differences('B4_parent128','B4_causal80',x,y)
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
