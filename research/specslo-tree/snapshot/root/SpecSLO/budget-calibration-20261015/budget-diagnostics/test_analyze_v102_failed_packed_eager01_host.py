"""Failure evidence parser controls; no actual failed-run audit is executed here."""
import copy
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
spec=importlib.util.spec_from_file_location('_V102_failed_HOST',P/'analyze_v102_failed_packed_eager01.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
spec=importlib.util.spec_from_file_location('_V102_complete_HOST_fixtures',P/'test_analyze_v102_actual_packed_eager01_cost_host.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


class FailureHOST(unittest.TestCase):
    def test_synthetic_packed18_retained_without_claiming_complete_controls_protocol_or_cost(self):
        r=h.report(0);r.update(control_numerical=[],protocol=[],fixed_point=[],timing=[],sealed=False)
        parsed=f.packed_records(r)
        self.assertEqual(len(parsed),6);self.assertEqual(sum(p['independent_numeric_records'] for p in parsed),18)
        self.assertFalse(parsed[-1]['metadata_scope']['C96_service_feasible'])

    def test_false_KV_or_incomplete_F_T_F_numeric_is_rejected(self):
        for kind in ('KV','missing'):
            r=h.report(0)
            if kind=='KV':r['numerical'][0]['all_layer_query_KV']=False
            else:r['numerical'].pop()
            with self.assertRaises(RuntimeError):f.packed_records(r)

    def test_primary_observer_ownership_failure_preserves_true_first_repair_owner7(self):
        r=h.report(0);r['warmup']=[dict(case=f.a.cost_cases()[0],method='original_fast4')]
        error=dict(stage='independent-same-M-allphysical-hidden-KV',error='prepare_control trace_observer.py padding_reference.py PA padding must preserve token0/pos0/slot-1/context1 and fresh owner0 table')
        parsed=f.classify_first_error(error,r)
        self.assertEqual(parsed['actual_repair_owner_ids'],list(range(7,13)))
        self.assertEqual((parsed['first_live_repair_owner'],parsed['repair_physical_M']),(7,8))
        self.assertFalse(parsed['candidate_or_Native_model_failure_proved'])

    def test_failure_mode_refuses_running_or_complete_before_other_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            w=Path(d)
            for status in ('running','complete'):
                (w/'cost-receipt-packed-eager01-HOST.json').write_text(json.dumps(dict(schema='V102.ragged01-packed-cost-receipt.v1',status=status)))
                with patch.object(f.a,'WORK',w),patch.object(f.a,'source_and_HOST',side_effect=AssertionError('too early')):
                    with self.assertRaisesRegex(RuntimeError,'actual terminal failed'):f.audit_failed('packed-eager01-HOST')


if __name__=='__main__':
    start=time.perf_counter();stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FailureHOST))
    row=dict(status='passed' if result.wasSuccessful() and 'torch' not in sys.modules else 'failed',tests=result.testsRun,
        failures=[dict(test=str(t),error=e) for t,e in result.failures],errors=[dict(test=str(t),error=e) for t,e in result.errors],
        actual_HOST_argv=[sys.executable,*sys.argv],elapsed_seconds=time.perf_counter()-start,
        input_hashes={str(p):f.a.sha(p) for p in (Path(__file__),P/'analyze_v102_failed_packed_eager01.py',P/'analyze_v102_actual_packed_eager01_cost.py',P/'test_analyze_v102_actual_packed_eager01_cost_host.py')},
        actual_failed_V102_audit_executed=False,torch_imported=('torch' in sys.modules),NPU_launched=False)
    (P/'V102_FAILED_FORENSICS_HOST_RESULT.json').write_text(json.dumps(row,indent=2)+'\n')
    print(stream.getvalue(),end='');print(json.dumps(row,indent=2));raise SystemExit(row['status']!='passed')
