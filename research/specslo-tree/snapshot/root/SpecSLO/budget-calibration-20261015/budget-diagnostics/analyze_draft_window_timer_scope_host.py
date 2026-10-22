"""Read-only HOST evidence about the existing draft window timer."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from collections import Counter

B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-s35-qualified-counts-v119'
OLD=B/'tree-worker-window-qualified-B5-v63'
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)


def execute_original_guard(name):
    functions=[n for n in ast.walk(ast.parse(NATIVE.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(functions)==1
    function=copy.deepcopy(functions[0]);function.decorator_list=[];function.returns=None
    for argument in [*function.args.posonlyargs,*function.args.args,*function.args.kwonlyargs]:argument.annotation=None
    if function.args.vararg:function.args.vararg.annotation=None
    if function.args.kwarg:function.args.kwarg.annotation=None
    class NoTorch:
        def __getattr__(self,name):raise AssertionError('Unexpected tensor/device API: '+name)
    scope={'torch':NoTorch(),'ProfiledRoofline':type('HOSTProfiledRoofline',(),{})}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
                 str(NATIVE)+':HOST-guard','exec'),scope)
    if name=='target_tree_forward':
        assert scope[name](SimpleNamespace(is_draft=True),[],[],[]) is None
    else:
        assert scope[name](SimpleNamespace(config=SimpleNamespace(spec_rhythm_roofline=None)),
                           None,has_target_work=True) is None
    return dict(original_function_body_executed=True,annotations_and_decorators_only_removed=True,
                tensor_or_collective_calls=0,return_value=None)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError('Fresh output required')
    files=[NATIVE,*[P/name for name in ('runtime_tree_decode.py','atomic_decode_patch.py',
        'live_runtime_hooks.py','original_hook_adapter.py','live_native_boundaries.py',
        'live_runtime_factory.py','original_chunk_binding.py','early_target.py','packed_window.py',
        'fast_tree_draft_v5.py','root_seed.py','startup_config_v75.py')]]
    before={str(p):sha(p) for p in files}
    source=(P/'runtime_tree_decode.py').read_text()
    anchors=('draft_started = time.perf_counter()', 'draft_output = self.draft_tree_forward',
        'target_residual_started = time.perf_counter()', 'target_context = _dt_submit_current_tree_target()',
        'target_submit_ended = target_context',
        'self._validate_spec_rhythm_target_graph(target_output', 'draft_ended = time.perf_counter()',
        'message = self._exchange_spec_rhythm_tree_candidates',
        'window_estimator.observe(draft_compute_ms=draft_ms')
    start=next(i for i,line in enumerate(source.splitlines(),1) if anchors[0] in line)
    lines={}
    for anchor in anchors:
        matches=[i for i,line in enumerate(source.splitlines(),1) if i>=start and anchor in line]
        assert len(matches)==1,(anchor,matches)
        lines[anchor]=matches[0]
    assert list(lines.values())==sorted(lines.values())
    patch=(P/'atomic_decode_patch.py').read_text()
    assert patch.count('_v75_hooks.prepare_chunk_at_original_target_join(local_states)')==1
    hook=(P/'original_hook_adapter.py').read_text()
    for stage in ('mixed-target-submit-at-original-join','original-native-chunk-finite-and-broadcast',
                  'transaction-prepared-host-commit'):assert stage in hook
    result=OLD/'measurement-output/runs/candidate-on-worker-window-qualified-rootseed-B5-full463-r1/result.json'
    receipt=OLD/'worker-window-qualified-rootseed-B5-full463-r1-receipt.json'
    old=load(result);r=old['results'][0];rec=load(receipt)
    assert rec['status']=='complete' and rec['errors']==[] and rec['experimental_sources_unchanged']
    for name in ('runtime_tree_decode.py','early_target.py','packed_window.py','fast_tree_draft_v5.py'):
        path=OLD/name
        assert sha(path)==rec['source_hashes'][str(path)]
        files.append(path)
    assert sha(result)==rec['result_sha256']
    assert old['spec_rhythm_roofline'] is None and old['profile_decode_steps']==old['profile_host_decode_steps']==0
    assert r['decode_timeline']==[] and r['decode_profile'] is None and r['decode_host_profile'] is None
    metrics=r['worker_aclgraph_metrics'];draft=next(m for m in metrics if m['rank']==0)
    target=next(m for m in metrics if m['rank']==1)
    rounds=draft['worker_spec_rhythm_tree_rounds']
    assert rounds==draft['dynamic_tree_measured_fast_draft_chain_calls']==target['worker_spec_rhythm_tree_early_target_target_cycles']==2379
    v96={};rawpaths=[]
    for rank in (0,1):
        path=B/f'tree-capacity64-runtime-v96/local-output/capacity64-full463-v1/rank{rank}-full463-actual-trace.json.gz'
        data=load(path);rawpaths.append(path)
        counts=Counter(event.get('stage') for event in data['boundary_events'])
        mixed=counts['mixed-target-submit-at-original-join']
        assert mixed==counts['original-draft-prompt-local-forward']==counts['original-native-chunk-finite-and-broadcast']==14
        assert data['decode_counters']['spec_rhythm_tree_rounds']==1491
        v96[str(rank)]=dict(rounds=1491,mixed_join_rounds=mixed,
            phase_seconds=data['phase_seconds'],eager_promoted=data['decode_counters']['spec_rhythm_tree_eager_promoted'],
            per_cycle_clock_or_window_samples_exported=False,
            raw_boundary_records_have_timestamps=False)
    guard_proofs={name:execute_original_guard(name) for name in
                  ('target_tree_forward','_validate_spec_rhythm_target_graph')}
    after={str(p):sha(p) for p in files if str(p) in before}
    assert after==before
    evidence=[result,receipt,*rawpaths,*files]
    data=dict(schema='draft-window-timer-scope.readonly-HOST.v1',
        created_UTC=datetime.now(timezone.utc).isoformat(),actual_HOST_argv=[sys.executable,*sys.argv],
        status='complete-readonly-scope-audit',source_and_raw_sha256={str(p):sha(p) for p in evidence},
        source_unchanged=True,original_guard_HOST_proofs=guard_proofs,base_source_anchor_lines=lines,
        V119_live_scope=dict(source_only=True,actual_current_run_artifacts_opened=False,
            timer='draft_started through draft_ended on rank0; passed verbatim through original proposal envelope',
            includes=['draft kwargs/own model forward/encoded D2H/existing local fence',
                'rank0 local target bookkeeping',
                'when mixed is planned: common submit-status join, real draft prompt forward, original finite vote/stream fence/Gloo barrier/final WORLD broadcast/transaction preparation',
                'draft candidate selection and original materialization/no-op after target join'],
            mixed_timer_is_pure_draft_compute=False,
            target_device_sync_on_other_ranks_alone_blocks_draft_rank=False,
            mixed_join_can_include_target_or_other_rank_wait=True,
            wait_milliseconds_or_unique_eager_failure_cause_proved=False),
        V63_actual=dict(result=str(result),rounds=rounds,worker_tree_rounds=rounds,
            outer_decode_rounds_including_other_rounds=r['decode_rounds'],
            profile_decode_steps=0,profile_host_decode_steps=0,decode_timeline_rows=0,
            rank0_draft_worker_seconds=draft['worker_draft_seconds'],
            rank0_draft_mean_ms=draft['worker_draft_seconds']*1000/rounds,
            rank0_local_target_bookkeeping_seconds=draft['worker_target_seconds'],
            rank0_local_target_bookkeeping_mean_ms=draft['worker_target_seconds']*1000/rounds,
            target_leader_worker_wall_mean_ms=target['worker_spec_rhythm_tree_early_target_total_wall_ns']/1e6/rounds,
            normal_bucket16_final_EMA_ms=draft['dynamic_tree_window_normal_bucket16_ms'],
            eager_bucket1_extra_EMA_ms=draft['dynamic_tree_window_eager_bucket1_overhead_ms'],
            eager_observations=draft['dynamic_tree_window_eager_observations'],
            eager_promoted=draft['worker_spec_rhythm_tree_eager_promoted'],
            actual_ProfiledRoofline=False,all4_mixed_join_installed=False,
            target_wait_inside_ordinary_rank0_draft_interval_established=False),
        V96_actual=v96,
        interpretation=['V63 final normal EMA close to target mean is not proof of target-wait pollution; the actual rank0 target/roofline guards return locally.',
            'V96 has14 mixed joins among1491 rounds, so mixed timer contamination is real in scope but its duration/final EMA effect is not recoverable from these aggregates.',
            'Whole-worker intervals contain HOST packing/D2H/selection and own model work. No isolated NPU compute times or overlap were measured.',
            'observe omits communication_ms, whose default is0. Exchange wall also includes skew/wait, so it must not be blindly subtracted as pure communication.'],
        minimal_future_measurement=dict(implementation_created=False,change_original_observe_input=False,
            scope='bounded unscored first warm65 only; detach before zero/full',
            outer_timestamp_points=['existing draft_started','before draft_tree_forward','after draft_tree_forward',
                'after existing draft synchronize','before/after original mixed join',
                'before candidate validation/selection','existing draft_ended','before/after existing proposal exchange'],
            mixed_stage_timestamp_points=['stage entry/callback entry/callback exit/stage exit per rank',
                'around existing native chunk finite vote and existing local stream fence; add no fence'],
            retain=['cycle/rank/normal+eager IDs/B vector/physical bucket/method/prompt chunk range',
                'legacy wall and its unchanged actual estimator observation','per-rank raw monotonic timestamps, finite/commit/exception order unchanged'],
            forbid=['profile_decode_steps, which disables early target and adds sync/profile collectives',
                'new tensor reads or clock/publication changes','calling all join wall target wait or all call wall NPU compute',
                'using final EMA versus aggregate target mean as a per-cycle gate proof']),
        NPU_started=False,inference_modules_imported=False,new_performance_or_root_cause_claim=False)
    with args.output.open('x') as stream:json.dump(data,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status=data['status'],V63_draft_mean_ms=data['V63_actual']['rank0_draft_mean_ms'],
        V63_rank0_target_mean_ms=data['V63_actual']['rank0_local_target_bookkeeping_mean_ms'],
        V96_mixed_join_rounds=14,NPU_started=False,output_sha256=sha(args.output))))


if __name__=='__main__':main()
