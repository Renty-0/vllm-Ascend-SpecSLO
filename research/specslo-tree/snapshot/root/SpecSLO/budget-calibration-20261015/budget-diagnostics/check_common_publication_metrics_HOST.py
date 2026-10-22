"""Regression checks for SLO denominators, empty final events, and token hashes."""
import copy
import json
from pathlib import Path
import unittest

from common_publication_metrics import compute, percentile, token_digest


def fixture():
    tokens = list(range(256))
    requests = [dict(request_id='request0', slo_class='tight', slo_tpot_ms=40)]
    events = [dict(request_id='request0', token_ids=tokens[:1], finished=False, clock=0.),
              dict(request_id='request0', token_ids=tokens[1:], finished=True, clock=10.24)]
    return requests, {'request0':tokens}, events


class MetricsTests(unittest.TestCase):
    def score(self, requests=None, outputs=None, events=None, **kwargs):
        r, o, e = fixture()
        return compute(r if requests is None else requests, o if outputs is None else outputs,
                       e if events is None else events, kwargs.pop('wall', 11),
                       clock_field='clock', method='HOST_fixture', expected_requests=1, **kwargs)

    def test_exact_paper_boundary_passes_while_observed_is_larger(self):
        result = self.score()
        self.assertEqual(result['rows'][0]['paper_tpot_ms'], 40)
        self.assertGreater(result['rows'][0]['observed_tpot_ms'], 40)
        self.assertEqual(result['all']['attained'], 1)
        self.assertEqual(result['all']['Goodput'], 256/11)

    def test_empty_finished_event_does_not_extend_TPOT(self):
        _, _, events = fixture()
        events[-1]['finished'] = False
        events.append(dict(request_id='request0', token_ids=[], finished=True, clock=10.9))
        result = self.score(events=events)
        self.assertEqual(result['rows'][0]['last_publication'], 10.24)
        self.assertEqual(result['rows'][0]['paper_tpot_ms'], 40)

    def test_original_final_hash_must_match_all_tokens(self):
        _, outputs, _ = fixture()
        expected = {'request0':dict(output_tokens=256, sha256=token_digest(outputs['request0']))}
        self.assertTrue(self.score(outputs=expected)['all_final_outputs_reconstructed'])
        expected['request0']['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'original final token'):
            self.score(outputs=expected)

    def test_changed_event_token_rejects_full_reconstruction(self):
        _, _, events = fixture()
        events[-1]['token_ids'][3] += 1
        with self.assertRaisesRegex(ValueError, 'original final tokens'):
            self.score(events=events)

    def test_missing_finish_and_post_finish_events_are_rejected(self):
        _, _, events = fixture()
        events[-1]['finished'] = False
        with self.assertRaises(ValueError):
            self.score(events=events)
        _, _, events = fixture()
        events.append(dict(request_id='request0', token_ids=[], finished=False, clock=10.8))
        with self.assertRaisesRegex(ValueError, 'after finish'):
            self.score(events=events)

    def test_nonmonotonic_and_nonfinite_clocks_are_rejected(self):
        _, _, events = fixture()
        events[0]['clock'] = 10.5
        with self.assertRaises(ValueError):
            self.score(events=events)
        for value in (float('nan'), float('inf'), True, -1):
            with self.assertRaises(ValueError):
                self.score(wall=value)

    def test_original_epoch_origin_does_not_change_interval(self):
        _, _, events = fixture()
        for event in events:
            event['clock'] += 100
        result = self.score(events=events, clock_origin=100)
        self.assertAlmostEqual(result['rows'][0]['paper_tpot_ms'], 40)
        with self.assertRaises(ValueError):
            self.score(events=events, clock_origin=101)

    def test_duplicate_request_ids_and_unknown_owners_are_rejected(self):
        r, o, e = fixture()
        with self.assertRaises(ValueError):
            compute(r+r, o, e, 11, clock_field='clock', method='HOST', expected_requests=2)
        e[0]['request_id'] = 'wrong'
        with self.assertRaises(ValueError):
            self.score(events=e)

    def test_linear_interpolated_percentile(self):
        self.assertEqual(percentile([40, 10, 20], .5), 20)
        self.assertEqual(percentile([40, 10, 20], .9), 36)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MetricsTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    output = Path(__file__).with_name('COMMON_PUBLICATION_METRICS_HOST_RESULT.json')
    with output.open('x') as stream:
        json.dump(dict(passed=result.wasSuccessful(), tests=result.testsRun,
                       real_device_or_model_operations=0), stream, indent=2)
        stream.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
