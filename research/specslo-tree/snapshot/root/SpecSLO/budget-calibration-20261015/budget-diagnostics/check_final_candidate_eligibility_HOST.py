"""HOST-only negative controls for admitting a real candidate to final repeats."""
import copy,json,unittest
from final_candidate_eligibility import qualified
from final_common_comparison import FIXED_FLOOR,OFFICIAL_FLOOR


def inputs():
    receipt=dict(status='complete',sources_unchanged=True,devices_after=dict(all_idle=True),owned_remaining=[])
    for phase in ('pre_verifier','post_verifier'):
        receipt[phase]=dict(exit_code=0,actual_result=dict(status='pass'))
    reference=dict(same_backend_exact_output_qualified=True,references_per_target=463,
        by_target_rank=[dict(rank=r,comparison=dict(requests=463,equal_requests=463,
            mismatching_requests=0,exact_all_tokens_equal=True))for r in (1,2,3)])
    audit=dict(status='terminal_completed_compute_audited',supervisor_status='complete',
               errors=[],whole_workload_AR_qualified=True,independent_complete_own_AR=reference)
    metrics=dict(all=dict(Goodput=650.0,Raw=650.0))
    return receipt,audit,metrics


class Eligibility(unittest.TestCase):
    def test_positive_is_only_eligibility_not_goal(self):
        result=qualified(*inputs())
        self.assertTrue(result['qualified_for_final_comparison'])
        self.assertFalse(result['goal_achieved'])
        self.assertTrue(result['final_repeats_still_required'])

    def rejected(self,change):
        args=inputs();change(*args)
        with self.assertRaises(ValueError):qualified(*args)

    def test_failed_guard_rejected(self):self.rejected(lambda r,a,m:r.update(status='failed'))
    def test_changed_source_rejected(self):self.rejected(lambda r,a,m:r.update(sources_unchanged=False))
    def test_leftover_worker_rejected(self):self.rejected(lambda r,a,m:r.update(owned_remaining=[17]))
    def test_busy_card_rejected(self):self.rejected(lambda r,a,m:r['devices_after'].update(all_idle=False))
    def test_failed_original_postcheck_rejected(self):
        self.rejected(lambda r,a,m:r['post_verifier']['actual_result'].update(status='failed'))
    def test_failed_independent_audit_rejected(self):self.rejected(lambda r,a,m:a.update(errors=['bad mask']))
    def test_AR_gate_rejected(self):self.rejected(lambda r,a,m:a.update(whole_workload_AR_qualified=False))
    def test_one_rank_disagrees_rejected(self):
        self.rejected(lambda r,a,m:a['independent_complete_own_AR']['by_target_rank'][1]['comparison'].update(
            equal_requests=462,mismatching_requests=1,exact_all_tokens_equal=False))
    def test_missing_rank_rejected(self):
        self.rejected(lambda r,a,m:a['independent_complete_own_AR']['by_target_rank'].pop())
    def test_only_462_requests_rejected(self):
        self.rejected(lambda r,a,m:a['independent_complete_own_AR'].update(references_per_target=462))
    def test_exact_1p5_floor_rejected(self):
        self.rejected(lambda r,a,m:m['all'].update(Goodput=1.5*OFFICIAL_FLOOR))
    def test_fixed_floor_rejected(self):
        self.rejected(lambda r,a,m:m['all'].update(Goodput=FIXED_FLOOR))
    def test_nonfinite_gp_rejected(self):self.rejected(lambda r,a,m:m['all'].update(Goodput=float('nan')))


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Eligibility)
    outcome=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(tests=outcome.testsRun,passed=outcome.wasSuccessful(),NPU_used=False,
                         real_candidate_selected=False,real_model_runs=0)))
    raise SystemExit(0 if outcome.wasSuccessful()else 1)
