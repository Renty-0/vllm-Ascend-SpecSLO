"""Read-only complete V96 HOST census; no device import/execution."""
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest

P=Path(__file__).resolve().parent
BASE=P.parent
RUN=BASE/'tree-capacity64-runtime-v96/local-output/capacity64-full463-v1'
FIXED=BASE/'serial-v1/paired-trial/measurement-output/runs/candidate-on-gamma4-current-cards-r1/result.json'
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py')
KEYS=('spec_rhythm_tree_eager_promoted','spec_rhythm_tree_eager_invalidated','spec_rhythm_tree_eager_dependency_rejections')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_trace(role,rank):
    path=RUN/f'rank{rank}-{role}-actual-trace.json.gz'
    return path,json.loads(gzip.decompress(path.read_bytes()))


def census(role):
    rows=[read_trace(role,rank) for rank in range(4)];traces=[r for path,r in rows]
    counters=[{key:trace['decode_counters'][key] for key in KEYS} for trace in traces]
    if any(value!=counters[0] for value in counters[1:]):raise ValueError('V96 four-rank eager counters differ')
    ordinary=[[(r['N'],r['budgets']) for r in trace['normal_target_calls']] for trace in traces[1:]]
    if any(value!=ordinary[0] for value in ordinary[1:]):raise ValueError('V96 three-target logical geometries differ')
    mixed=[[(len(r['target_ids']),r['budgets']) for r in trace['logical_events'] if r['kind']=='plan'] for trace in traces]
    if any(value!=mixed[0] for value in mixed[1:]):raise ValueError('V96 four-rank mixed geometries differ')
    calls=ordinary[0]+mixed[0];n32=[(n,b) for n,b in calls if n==32]
    if len(calls)!=traces[1]['decode_counters']['spec_rhythm_tree_rounds']:
        raise ValueError('Real target record census does not close original rounds')
    if sum(sum(b) for n,b in calls)!=traces[1]['decode_counters']['spec_rhythm_verified_tokens']:
        raise ValueError('Real selected candidate census does not close original verified count')
    histogram=Counter((n,sum(b)) for n,b in calls)
    return dict(role=role,source_generation=traces[1]['source_generation'],raw_sha256={str(path):sha(path) for path,trace in rows},
        all_four_rank_eager_counters_identical=True,three_target_logical_geometries_identical=True,
        counters=counters[0],target_calls=len(calls),ordinary_target_calls=len(ordinary[0]),mixed_target_calls=len(mixed[0]),
        verified_candidates=sum(sum(b) for n,b in calls),C96_target_calls=sum(sum(b)==96 for n,b in calls),
        N32_target_calls=len(n32),N32_C96_calls=sum(sum(b)==96 for n,b in n32),
        N32_max_B_histogram=dict(Counter(max(b) for n,b in n32)),
        selected_B_histogram=dict(Counter(value for n,b in calls for value in b)),
        N32_selected_B_histogram=dict(Counter(value for n,b in n32 for value in b)),
        uniform_B5_target_groups=sum(set(b)=={5} for n,b in calls),
        uniform_B5_groups_N_le18=sum(set(b)=={5} and n<=18 for n,b in calls),
        common_N_C=[dict(N=n,C=c,calls=count) for (n,c),count in histogram.most_common(12)],
        actual_controller_candidates=None,actual_gap_urgency_acceptance_extent_counts=None,
        actual_window_offers_selection=None,actual_policy_need_grants=None,actual_generated=None,actual_published=None,
        verified_proposal_group_is_concurrent_draft_normal_grant_claimed=False,
        measured_pack_path_occurrences=None,old_V67_unsupported_counts_used=False)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    sources=[Path(__file__),P/'rolling_eager_low_budget_host.py',P/'test_rolling_eager_low_budget_host.py',NATIVE,
        NATIVE.with_name('spec_rhythm.py'),NATIVE.with_name('tree.py'),
        BASE/'tree-s35-production-v101/PREPARED_SOURCE_HASHES_V101.json',
        *[BASE/'tree-s35-production-v101'/name for name in ('budget_policy.py','tree_entry.py','runtime_tree_decode.py','packed_window.py','fast_tree_draft_v5.py','backbone_selection.py')],
        *[BASE/'tree-eager-shadow-overlay-v99'/name for name in ('PARENT_BINDING_V99.json','PATCH_PLAN_V99.json','eager_overlay_v99.py','runtime_patch_v99.py')],
        BASE/'tree-packed-eager01-cost-v102/packed_root_seed_eager3.py',FIXED]
    before={str(path):sha(path) for path in sources}
    result=dict(schema='V96.rolling-eager-complete-HOST-census-and-uninstalled-prototype.v1',UTC=datetime.now(timezone.utc).isoformat(),
        argv=sys.argv,python=sys.executable,python_version=sys.version,runtime_wrapper_invoked=False,
        input_sha256=before,roles=[census(role) for role in ('warmup65','zero65-1','zero65-2','full463')])
    fixed=json.loads(FIXED.read_text());workers=fixed['results'][0]['worker_aclgraph_metrics']
    names=('worker_spec_rhythm_eager_proposals','worker_spec_rhythm_eager_promoted','worker_spec_rhythm_eager_invalidated',
        'worker_spec_rhythm_fixed_gamma_scheduler_fast_path','worker_spec_rhythm_fixed_gamma_scheduler_fast_path_cycles',
        'worker_spec_rhythm_slo_home_partition_home0_capacity','worker_spec_rhythm_slo_home_partition_home1_capacity')
    metrics=[{name:r[name] for name in names} for r in workers]
    if any(row!=metrics[0] for row in metrics[1:]):raise ValueError('Fixed gamma actual four-worker counters differ')
    result['fixed_gamma_actual']=dict(path=str(FIXED),sha256=sha(FIXED),
        metric_path='results[0].worker_aclgraph_metrics[rank]',metrics=metrics[0],metrics_are_generation_fields_not_measured_graph_delta=True,
        config={key:fixed[key] for key in ('gamma','spec_rhythm_linear_full_window','spec_rhythm_min_gamma','spec_rhythm_max_eager_tokens',
            'spec_rhythm_verification_budget','spec_rhythm_draft_token_budget','spec_rhythm_eager_reserve_tokens','spec_rhythm_roofline',
            'spec_rhythm_linear_eager_cross_graph_bucket','spec_rhythm_linear_idle_residual_eager')},
        source_roof_rule='active_rows*gamma; active64/gamma4 gives256, not tree combinedC96',
        direct_performance_or_promotion_ratio_comparison_qualified=False)
    transcript=io.StringIO();outcome=unittest.TextTestRunner(stream=transcript,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('test_rolling_eager_low_budget_host'))
    from rolling_eager_low_budget_host import prepared_selector_source
    result['prototype']=prepared_selector_source()[1]
    result['HOST_tests']=dict(tests_run=outcome.testsRun,failures=len(outcome.failures),errors=len(outcome.errors),transcript=transcript.getvalue())
    after={name:sha(name) for name in before}
    raw_before={name:value for row in result['roles'] for name,value in row['raw_sha256'].items()}
    raw_after={name:sha(name) for name in raw_before}
    result.update(input_sha256_after=after,all_source_and_actual_raw_unchanged=before==after and raw_before==raw_after,
        Torch_imported='torch' in sys.modules,NPU_started=False,installed=False,numerical_qualification=False,formal_Goodput=False)
    result['passed']=outcome.wasSuccessful() and result['all_source_and_actual_raw_unchanged'] and not result['Torch_imported']
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(passed=result['passed'],tests_run=outcome.testsRun,output=str(args.output),sha256=sha(args.output),
        actual_full463_promoted=result['roles'][-1]['counters'][KEYS[0]],NPU_started=False)))
    return int(not result['passed'])


if __name__=='__main__':raise SystemExit(main())
