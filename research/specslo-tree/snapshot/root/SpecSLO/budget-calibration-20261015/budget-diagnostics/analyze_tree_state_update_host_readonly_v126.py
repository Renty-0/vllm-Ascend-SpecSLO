"""Read existing cProfile/results and frozen source only; stdlib, no model imports."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pstats
import sys

B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
V=B/'tree-s35-qualified-counts-v119';C=B/'tree-worker-window-qualified-B5-v63'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parent_path=D/'V103_ACTUAL_FIRST_WARMUP_HOST_PROFILE.json'
    parent=json.loads(parent_path.read_text());profile_path=Path(parent['profile']['path'])
    assert sha(profile_path)==parent['profile']['sha256']
    stats=pstats.Stats(str(profile_path)).stats
    def function(key):
        cc,nc,tt,ct,callers=stats[key]
        return dict(file=key[0],line=key[1],function=key[2],primitive_calls=cc,calls=nc,
            exclusive_wall_seconds=tt,cumulative_wall_seconds=ct,
            callers=[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],calls=v[1],
                exclusive_wall_seconds=v[2],cumulative_wall_seconds=v[3])
                for k,v in sorted(callers.items(),key=lambda x:-x[1][3])[:12]])
    names={'_preflight_spec_rhythm_tree_cache_commit','add_mapping','_spec_rhythm_nonfinite_flag',
        '_tree_primary_path_length','tree_primary_path','finish_verification','validate_verification',
        'apply_tree_verification','_deliver_committed_tokens','_tree_row_requires_kv_compaction',
        'add_decode_time','snapshot','_pages','_tables','_check_binding','_verify_tree_outputs_host',
        'target_tree_forward','make_host_tree_attention_metadata','_update_attention_task_list',
        '_update_fused_infer_attention_task'}
    selected=[function(k) for k in sorted(stats,key=lambda k:-stats[k][3])
        if k[2] in names or (k[2] in ('broadcast','reduce','barrier') and 'host_tree_transport.py' in k[0])
        or (k[2]=='__torch_function__' and '/torch/utils/_device.py' in k[0])]
    baseline_paths=dict(V63=C/'measurement-output/runs/candidate-on-worker-window-qualified-rootseed-B5-full463-r1/result.json',
        fixed=B/'serial-v1/paired-trial/measurement-output/runs/candidate-on-gamma4-current-cards-r1/result.json')
    baselines={}
    for label,path in baseline_paths.items():
        v=json.loads(path.read_text());r=v['results'][0];phase=r['decode_phase_seconds'];rounds=r['decode_rounds']
        row_rounds=sum(r['request_verification_rounds']);state=phase['state_update'];e2e=r['e2e_elapsed_seconds']
        baselines[label]=dict(path=str(path),sha256=sha(path),decode_phase_seconds=phase,outer_rounds=rounds,
            summed_request_verification_rounds=row_rounds,state_ms_per_outer_round=state*1000/rounds,
            state_us_per_request_verification_round=state*1e6/row_rounds,e2e_elapsed_seconds=e2e,
            phase_state_fraction_of_e2e=state/e2e,
            ideal_arithmetic_speedup_if_entire_phase_vanished=e2e/(e2e-state),
            entire_phase_removal_is_not_a_real_algorithm_or_Goodput_prediction=True,
            detailed_host_profile_available=r['decode_host_profile'] is not None)
    sources=[C/'runtime_tree_decode.py',V/'runtime_tree_decode.py',V/'atomic_decode_patch.py',
        V/'original_hook_adapter.py',V/'live_runtime_hooks.py',V/'native_boundaries.py',
        V/'host_tree_transport.py',V/'combined_controls_v113.py',V/'cpu_default_device_cleanup_host.py',
        V/'fast_tree_draft_v5.py',V/'root_seed.py',N/'native_engine.py',N/'tree.py',N/'spec_rhythm.py',N/'native_model.py',
        Path('/root/miniconda3/envs/vllm-hust-dev/lib/python3.11/site-packages/torch/utils/_device.py')]
    source_pins={str(path):sha(path) for path in sources}
    output=dict(schema='V126.existing-tree-state-update-readonly-HOST-audit.v1',
        UTC=datetime.now(timezone.utc).isoformat(),actual_HOST_argv=list(sys.orig_argv),
        analyzer_sha256=sha(Path(__file__)),parent_analysis=dict(path=str(parent_path),sha256=sha(parent_path)),
        actual_profile=dict(parent['profile']),profiled_original_generate_seconds=parent['actual_profiled_original_generate_seconds'],
        profiled_normal_calls=parent['actual_normal_target_calls'],profiled_phase_seconds=parent['phase_seconds'],
        profiled_state_ms_per_normal_call=parent['phase_seconds']['state_update']*1000/parent['actual_normal_target_calls'],
        selected_actual_functions=selected,
        actual_top_exclusive=[function(k) for k in sorted(stats,key=lambda k:-stats[k][2])[:18]],
        baselines=baselines,source_pins=source_pins,
        V63_and_V119_base_runtime_bytes_equal=sha(C/'runtime_tree_decode.py')==sha(V/'runtime_tree_decode.py'),
        boundaries=dict(tree_base_state_lines=[842,980],fixed_Native_state_lines=[12533,12663],
            tree_only_included=['ancestor/prefix/verdict structural checks','KV mapping preflight',
                'actual device finite flags copied to CPU and Gloo commit consensus','KV compaction',
                'dual draft+target state commits','exact eager dependency and promotion', 'finish_cycle'],
            tree_base_excluded=['CPU verifier','verdict broadcast','arrival/prefill/cycle clocks',
                'add_decode_time accounting','online admission/release'],
            V119_additional_inside_state='atomic_decode_patch inserts commit_after_original_finish_cycle before phase end; planned mixed chunk adds physical fence, all-rank consensus and cursor/known-KV ledger transition',
            fixed_mainly='full-window state/controller bookkeeping, eager payload and finish_cycle; no equivalent tree KV preflight/commit-consensus block'),
        limitations=['rank1 main thread only; draft rank0 FastV5 absent',
            'first warmup65, not full463 or current V119 combined arm',
            'cProfile measures elapsed wall, including C calls and synchronization waits',
            'cumulative values overlap and cannot be summed',
            '15808 update calls = 494*32 main-thread tasks with 2 workers, not a 32-layer model',
            'target CPU context cleanup already selected in combined; old target mode totals cannot be reapplied as new draft savings',
            'source or HOST operation count does not establish elapsed or Goodput gain'],
        new_NPU_or_Torch_import=False,frozen_sources_modified=False,device_cost_gain_claimed=False)
    out=D/'TREE_STATE_UPDATE_CPU_READONLY_AUDIT_V126.json'
    with out.open('x') as stream:json.dump(output,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(output=str(out),sha256=sha(out),profile_sha256=sha(profile_path),
        baselines=baselines,profiled_state_ms_per_normal_call=output['profiled_state_ms_per_normal_call']),indent=2))

if __name__=='__main__':main()
