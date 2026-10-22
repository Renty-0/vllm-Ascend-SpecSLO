"""Historical reference audit and future read-only rebind specification only."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=D/'final-baseline-pairing-prep-v1'
F=B/'serial-v1/paired-trial'
M=Path('/root/SpecSLO/upstream-baseline-20261014')
FIXED=F/'measurement-output/runs/candidate-on-gamma4-current-cards-r1/result.json'
STRONG=M/'runs/mc2-on-default-hccl-r1/result.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def require(v,msg):
    if not v:raise RuntimeError(msg)


def audit(path,kind):
    d=json.loads(path.read_text());r=d['results'][0];manifest=Path(d['request_manifest']);rows=[json.loads(s) for s in manifest.read_text().splitlines() if s]
    require(sha(manifest)==d['request_manifest_sha256']=='dc58dc5af1c5b28c6944f8f3faadeaae2dae34298c07b3ac84745d367c67d662','original manifest')
    require(len(rows)==len(r['output_token_ids'])==len(r['slo']['request_metrics'])==463 and all(len(row)==256 for row in r['output_token_ids']) and r['output_tokens']==118528,'all463x256')
    require(r['prompt_token_ids_sha256']=='6f543c04a4580ae9363c4469504847a708b277d13eb708f71975887389f21815','identical pretokenized inputs')
    errors=[];attained=Counter()
    for source,row in zip(rows,r['slo']['request_metrics'],strict=True):
        require(row['request_id']==source['request_id'] and row['slo_class']==source['slo_class'] and row['slo_tpot_ms']==source['slo_tpot_ms'] and row['output_tokens']==256,'request identity/SLO')
        ms=(row['finished_decode_elapsed_ms']-row['decode_start_elapsed_ms'])/256 if kind=='fixed' else (row['last_token_ts']-row['first_token_ts'])*1000/256
        require(math.isfinite(ms) and ms>=0,'raw perrequest time')
        errors.append(abs(ms-row['paper_tpot_ms']))
        if ms<=source['slo_tpot_ms']:attained[source['slo_class']]+=1
    gp=sum(attained.values())*256/r['e2e_elapsed_seconds'];raw=118528/r['e2e_elapsed_seconds']
    require(max(errors)<1e-9 and math.isclose(gp,r['slo']['paper']['goodput_tokens_per_e2e_second'],rel_tol=1e-12),'independent TPOT/Goodput')
    return d,r,dict(result=str(path),result_sha256=sha(path),Goodput=gp,raw_throughput=raw,e2e_seconds=r['e2e_elapsed_seconds'],
        attained_by_class=dict(attained),attained_requests=sum(attained.values()),max_raw_TPOT_recompute_error_ms=max(errors),
        original463_manifest_path=str(manifest),manifest_sha256=sha(manifest),arrival_seed=0,
        manifest_class_counts=dict(Counter(row['slo_class'] for row in rows)),
        manifest_class_SLO_ms={cls:sorted({row['slo_tpot_ms'] for row in rows if row['slo_class']==cls}) for cls in ('tight','normal','loose')},
        paper_TPOT_denominator=256,paper_TPOT_clock='Native cumulative finished_decode_elapsed-start' if kind=='fixed' else 'host monotonic last_token_ts-first_token_ts',
        e2e_timing_scope=r['e2e_timing_scope'],warmup_excluded=d['warmup_excluded_from_measurement'],
        output_exact_AR_equivalence_qualified=False)


def main():
    started=time.perf_counter();fd,fr,fixed=audit(FIXED,'fixed');md,mr,strong=audit(STRONG,'strong')
    receipt=F/'gamma4-current-cards-r1-receipt.json';frec=json.loads(receipt.read_text())
    require(frec['status']=='complete' and frec['errors']==[] and frec['Goodput']==fixed['Goodput'] and frec['result_sha256']==sha(FIXED)
        and frec['physical_cards']==[2,5,6,7] and frec['owned_remaining']==[],'historical fixed cleanup/source receipt')
    env=M/'runs/mc2-on-default-hccl-r1/launch-environment.json';menv=json.loads(env.read_text());evidence=M/'runs/mc2-on-default-hccl-r1/model-evidence.json';model=json.loads(evidence.read_text())
    require(len(model)==4 and all(row['enabled'] is True and row['selected_modules']==128 and row['projection_counts']=={'o_proj':64,'down_proj':64} and row['quantization'] is None for row in model),'strong actual resolved MC2 classes')
    require(menv['physical_devices']==[1,2,3,5] and fd['enable_mc2'] is True and fr['mc2_measured_replay_coverage']['passed'],'actual historical resources/MC2')
    fixed.update(physical_cards=frec['physical_cards'],draft_TP=1,target_TP=3,gamma=4,actual_protocol='dual-batch overlap with rolling eager; directory serial-v1 is not a serial-execution assertion',
        Native_source_commit=(FIXED.parent/'git-head.txt').read_text().strip(),working_tree_patch_sha256=sha(FIXED.parent/'working-tree.patch'),
        frozen_baseline_lock=str(Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/lock.json')),
        runtime='retained Python3.11, CANN9.0 + ATB9.0 two-shell runtime',
        historical_standalone_package_snapshot_present=False,
        runtime_environment=fd['runtime_environment'],MC2_profile=fr['mc2_measured_replay_coverage']['profile'],
        MC2_target_rank_observations=[{k:row[k] for k in ('rank','measured_replay_token_row_observations','qualified_full_chain_replay_count','qualified_mc2_replay_count','measured_mc2_fallback','measured_mc2_exception','static_route_count','passed')} for row in fr['mc2_measured_replay_coverage']['target_ranks']])
    strong.update(physical_cards=menv['physical_devices'],target_TP=4,dtype='bfloat16',packages=menv['packages'],python=menv['python'],
        official_vllm_commit=menv['vllm_commit'],official_ascend_commit=menv['ascend_commit'],runtime='base Python3.12/CANN9.1',
        HCCL_OP_EXPANSION_MODE=menv['environment']['HCCL_OP_EXPANSION_MODE'],LD_PRELOAD=menv['environment'].get('LD_PRELOAD'),
        actual_MC2_selected_modules_per_rank=[r['selected_modules'] for r in model],
        MC2_scope='actual resolved worker classes plus pinned apply_impl branch to torch_npu.npu_mm_all_reduce_base; no per-device timed kernel census',
        Native_TP3_exact22_row_MC2_contract_does_not_apply=True)
    names=[FIXED,STRONG,receipt,env,evidence,M/'runs/mc2-on-default-hccl-r1/independent-audit.json',M/'comparison.json',
        M/'measurement-source-lock.json',M/'device-mapping.json',M/'official-source-verification.json',M/'native-artifact-reuse.json',
        M/'run.sh',M/'run-measured-v1.sh',M/'preflight_current_devices.py',M/'audit_result.py',M/'harness/tp4_model_evidence.py',
        M/'plugin/vllm_ascend/ops/linear_op.py',M/'harness/run_baseline.py',M/'harness/examples/benchmark_nano_pearl_target_only.py',
        F/'run.sh',F/'body.sh',F/'supervise_trial.py',FIXED.parent/'mechanism-gate.json',FIXED.parent/'contract.env',
        FIXED.parent/'git-head.txt',FIXED.parent/'git-status.txt',FIXED.parent/'working-tree.patch',B/'serial-v1/runtime.sh',B/'serial-v1/map-current.py',
        Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/lock.json')]
    sources={str(p):sha(p) for p in names}
    spec=dict(schema='HOST.future-final-baseline-readonly-rebind-spec.v1',desired_physical_cards=[2,5,6,7],
        current_expected_user_ids=[0,1,2,3],actual_ACL_mapping_not_queried_by_this_HOST_preparation=True,
        original_sources_untouched=True,source_hashes=sources,
        fixed_command=['bash',str(F/'run.sh'),str(P/'future-results/fixed'),'final-fixed-r1'],
        fixed_existing_launcher_accepts_new_output_root=True,fixed_original_supervisor_writes_old_directory_and_must_not_be_used_directly=True,
        TP4_original_run_sh_rejects_changed_physical_mapping=True,TP4_original_measured_script_hardcodes_historical_physical_metadata=True,
        TP4_rebind_required_before_actual_launch=True,
        permitted_TP4_new_launcher_only_delta=[
            dict(file=str(M/'run-measured-v1.sh'),before='run=$root/runs/$tag',after='run=${SPECSLO_FINAL_RESULT_ROOT:?fresh private result root required}/runs/$tag'),
            dict(file=str(M/'run-measured-v1.sh'),before="'physical_devices':[1,2,3,5],'device_mapping':json.loads((root/'device-mapping.json').read_text()),",
                after="'physical_devices':[2,5,6,7],'device_mapping':json.loads(pathlib.Path(os.environ['SPECSLO_FINAL_MAPPING']).read_text()),"),
            dict(file=str(M/'preflight_current_devices.py'),before='expected = json.loads((root / "device-mapping.json").read_text())["devices"]',
                after='expected = json.loads(Path(FRESH_MAPPING_PATH).read_text())["devices"]')],
        TP4_python='/usr/local/python3.12.13/bin/python3',TP4_original_harness=str(M/'harness/run_baseline.py'),
        TP4_original_inference_arguments_and_flags_must_remain_exact=True,
        TP4_original_plugin_and_vllm_sources_readonly=True,
        future_actual_mapping_requirements=['fresh ACL9.1 user/logical/physical query validates all4 desired cards','actual lib SHA and current loaded driver/runtime metadata',
            'UTC/raw per-card idle snapshots before and after every run','own process cleanup and no unknown concurrent workers'],
        common_stream_TPOT_gap='unchanged fixed result has Native cumulative raw start/finish, not host first/last publication timestamp; add separately source-bound pureHOST delivery witness if common stream comparison required; do not relabel historical clocks',
        prepared_TP4_rebind_launcher_already_device_ready=False)
    plan=dict(schema='HOST.future-three-block-final-pair-plan.v1',source_bound_before_any_run=True,actual_NPU_launch_authorized_by_this_file=False,
        desired_physical_cards=[2,5,6,7],blocks=[dict(round=1,order=['fixed_gamma4','tree_final','strong_MC2_TP4']),dict(round=2,order=['strong_MC2_TP4','tree_final','fixed_gamma4']),dict(round=3,order=['fixed_gamma4','tree_final','strong_MC2_TP4'])],
        total_complete_independent_process_runs=9,paired_contrasts_per_reference=3,tree_run_shared_by_two_within_block_ratios=True,
        rank_count_is_not_trial_count=True,all_runs_original463_seed0_online_output256=True,
        same_card_results_must_be_new_not_copied_from_history=True,
        report_all_runs_all3_paired_ratios_and_median_no_posthoc_fastest_selection=True,
        historical_numerical_targets=dict(fixed_GP=501.63560407964184,strong_MC2_GP=429.8442987424407,strict_tree_GT_fixed=True,tree_GT_1_5x_strong=644.766448113661),
        future_final_tree_source_and_launcher_not_yet_selected=True,tree_restart_requires_own_warm_zero_seal_not_historical_flags=True,
        per_run_gate=['source+runtime+physical mapping and original full workload identities','actual matching SLO/paper denominator256 and timing source',
            'no perf profiler/heavy audit during timed interval','complete all463x256 and result SHA','sameBFruntimeprecision/MC2 actual role-specific evidence','cleanup, lock verifier and all4idle'],
        ratio_scopes_separate=['historical goal thresholds','new same-card legacy-paper GP ratios','new same-card common host stream GP ratios if witnessed'],
        alternate_order_must_not_be_changed_after_results=True)
    for name,value in [('REQUIRED_READONLY_TP4_REBIND.json',spec),('FINAL_THREE_BLOCK_PAIRING_PLAN.json',plan)]:
        with (P/name).open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    out=dict(schema='HOST.final-reference-origin-TPOT-and-resource-audit.v1',actual_HOST_argv=sys.orig_argv,status='complete-HOST-audit-only',
        fixed=fixed,strong_MC2=strong,current_goal_numerical_floor=strong['Goodput']*1.5,
        historical_strong_and_fixed_are_same_card_pair=False,same_card_final_1_5x_qualified=False,
        same_class_counts_and_paper_N_denominator=True,raw_clock_implementations_identical=False,
        MIGRATION_top_level_not_authoritative_for_current_best_cards_or_current_run=True,
        source_hashes=sources,configuration_files={str(P/x):sha(P/x) for x in ('REQUIRED_READONLY_TP4_REBIND.json','FINAL_THREE_BLOCK_PAIRING_PLAN.json')},
        no_NPU_query_or_model_initialized=True,torch_imported='torch' in sys.modules,elapsed_seconds=time.perf_counter()-started)
    require(not out['torch_imported'] and all(sha(p)==v for p,v in sources.items()),'read-only HOST closure')
    result=D/'FINAL_BASELINE_REFERENCE_ORIGIN_AND_PAIRING_HOST_AUDIT.json'
    with result.open('x') as stream:stream.write(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(path=str(result),sha256=sha(result),fixed=fixed['Goodput'],strong=strong['Goodput'],floor=strong['Goodput']*1.5,
        fixed_TPOT_error_ms=fixed['max_raw_TPOT_recompute_error_ms'],strong_TPOT_error_ms=strong['max_raw_TPOT_recompute_error_ms'],elapsed_seconds=out['elapsed_seconds'])))


if __name__=='__main__':main()
