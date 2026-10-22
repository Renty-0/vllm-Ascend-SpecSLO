"""Synthetic HOST controls only. Never opens a V119 actual runtime artifact."""
import copy
import hashlib
from pathlib import Path
import tarfile
import tempfile
import unittest

import analyze_v119_actual_combined as audit


class ExternalAuditorHOST(unittest.TestCase):
    def population(self):
        rows=[dict(request_id='HOST-'+name,slo_class=name,slo_tpot_ms=slo) for name,slo in audit.SLOS.items()]
        results=[];events=[]
        for i,row in enumerate(rows):
            tokens=[i]*256
            results.append(dict(request_id=row['request_id'],completion_token_ids=tokens,paper_tpot_ms=10.,
                observed_tpot_ms=10.*256/255,paper_slo_attained=True,paper_slo_goodput_tokens=256))
            events.extend([dict(request_index=i,request_id=row['request_id'],token_ids=tokens[:1],elapsed_seconds=float(2*i),finished=False),
                dict(request_index=i,request_id=row['request_id'],token_ids=tokens[1:],elapsed_seconds=float(2*i+1),finished=True)])
        return dict(generate_results=results,stream_events=events,host_generate_call_seconds=6.,elapsed_seconds=6.,native_elapsed_seconds=5.9),rows

    def test_independent_Native_and_stream_GP_are_distinct_and_json_scalars(self):
        trace,rows=self.population();value=audit.output_metrics(trace,rows)
        self.assertEqual(value['native_paper']['attained'],3)
        self.assertEqual(value['native_paper']['goodput_tokens_per_second'],128.)
        self.assertAlmostEqual(value['event_wall_paper']['median_tpot_ms'],1000/256)
        self.assertAlmostEqual(value['event_wall_observed']['median_tpot_ms'],1000/255)
        self.assertFalse(value['Native_TPOT_independently_reconstructed_from_raw_finish_start'])
        self.assertTrue(value['stream_event_wall_independently_recomputed'])

    def test_output_stream_token_clock_and_native_denominator_tampering_rejected(self):
        original,rows=self.population()
        for change in ('tokens','denominator','clock','terminal'):
            trace=copy.deepcopy(original)
            if change=='tokens':trace['generate_results'][0]['completion_token_ids'][0]+=1
            elif change=='denominator':trace['generate_results'][0]['observed_tpot_ms']+=1
            elif change=='clock':trace['stream_events'][1]['elapsed_seconds']=-1.
            else:trace['stream_events'][0]['finished']=True
            with self.subTest(change=change),self.assertRaises(ValueError):audit.output_metrics(trace,rows)

    def test_absolute_archive_member_bytes_and_missing_member_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.py';source.write_text('HOST = True\n')
            expected={str(source):audit.sha(source)};archive=Path(directory)/'source.tar.gz'
            with tarfile.open(archive,'w:gz') as output:output.add(source,arcname=str(source).lstrip('/'))
            result=audit.archive(archive,expected);self.assertEqual(result['members'],1)
            with self.assertRaises(ValueError):audit.archive(archive,{str(source):'a'*64})
            with self.assertRaises(ValueError):audit.archive(archive,{**expected,str(source)+'missing':'a'*64})

    def test_proposals_use_raw_before_after_delta_and_report_actual_FULL_mix(self):
        plans=[dict(N=2,budgets=[4,5]),dict(N=1,budgets=[3])]
        logical=[dict(kind='plan',target_ids=[1,2],budgets=[1,2])]
        eager={key:0 for key in ('spec_rhythm_tree_eager_promoted','spec_rhythm_tree_eager_invalidated','spec_rhythm_tree_eager_dependency_rejections')}
        records=[dict(normal_target_calls=[],logical_events=logical,decode_counters=eager)]
        records += [dict(normal_target_calls=plans,logical_events=logical,decode_counters=dict(eager,
            spec_rhythm_target_full_tree_rounds=2,spec_rhythm_target_causal_tree_rounds=1)) for _ in range(3)]
        expected=dict(target_branching_proposals=1,target_linear_proposals=4,**{f'actual_target_B{b}':1 for b in range(1,6)})
        envelopes=[dict(actual_proposal_counts=dict(before=dict.fromkeys(expected,500),after={k:500+v for k,v in expected.items()},delta=expected)) for _ in range(4)]
        result=audit.proposal_metrics(records,envelopes)
        self.assertEqual(result['B1_to_B5'],{str(b):1 for b in range(1,6)})
        self.assertEqual(result['branching_proposal_fraction'],.2)
        self.assertEqual(result['all_target_call_FULL_fraction'],2/3)
        envelopes[2]['actual_proposal_counts']['after']['actual_target_B5']+=1
        with self.assertRaises(ValueError):audit.proposal_metrics(records,envelopes)


if __name__=='__main__':unittest.main()
