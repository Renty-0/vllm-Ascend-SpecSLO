"""Read only the completed, SHA-bound first-warmup cProfile artifact.

cProfile records wall time in Python/C calls, including waits. Cumulative
times overlap and are not a CPU-vs-device decomposition or full-load score.
"""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pstats

BASE=Path('/root/SpecSLO/budget-calibration-20261015')
DIRECTORY=BASE/'tree-active-cap-controls-v103/local-output/active32-64-cap3-4-full463-v1'
TRACE=DIRECTORY/'active32-cap3/rank1-warmup65-actual-trace.json.gz'
OUT=BASE/'budget-diagnostics/V103_ACTUAL_FIRST_WARMUP_HOST_PROFILE.json'


def sha(blob):return hashlib.sha256(blob).hexdigest()


def main():
    if not TRACE.exists():
        raise SystemExit('Actual completed first-warmup trace not yet available')
    raw=TRACE.read_bytes();trace=json.loads(gzip.decompress(raw));artifact=trace['actual_HOST_profile']
    assert trace['rank']==1 and trace['role']=='warmup65' and trace['case']=='active32-cap3'
    assert trace['scope_count']==65
    path=Path(artifact['path']);assert path.resolve().is_relative_to(DIRECTORY.resolve())
    blob=path.read_bytes();assert sha(blob)==artifact['sha256'] and len(blob)==artifact['bytes']<=16*1024**2
    assert artifact['C_wait_wall_included'] and not artifact['pure_CPU_cycles_claimed'] and not artifact['full463_profile_enabled']
    stats=pstats.Stats(str(path))
    functions=[]
    for (file,line,name),(primitive,calls,exclusive,cumulative,callers) in stats.stats.items():
        functions.append(dict(file=file,line=line,function=name,primitive_calls=primitive,calls=calls,
            exclusive_wall_seconds=exclusive,cumulative_wall_seconds=cumulative,
            exclusive_us_per_call=exclusive*1e6/calls if calls else None,
            cumulative_us_per_call=cumulative*1e6/calls if calls else None))
    exclusive=sorted(functions,key=lambda r:r['exclusive_wall_seconds'],reverse=True)
    cumulative=sorted(functions,key=lambda r:r['cumulative_wall_seconds'],reverse=True)
    relevant=[row for row in functions if any(term in row['function'] for term in (
        '_validate_tree_fia_mask','_update_attention_task_list','_update_fused_infer_attention_task',
        'target_tree_forward','_broadcast_spec_rhythm_tree','_generate_spec_rhythm_tree',
        'make_tree_attention_metadata','_preflight_spec_rhythm','wait','synchronize'))]
    result=dict(schema='ROOT.V103.actual-warmup-only-cProfile-analysis.v1',UTC=datetime.now(timezone.utc).isoformat(),
        source_script_sha256=sha(Path(__file__).read_bytes()),trace=dict(path=str(TRACE),sha256=sha(raw)),
        profile=artifact,actual_profiled_original_generate_seconds=trace['host_generate_call_seconds'],
        actual_normal_target_calls=len(trace['normal_target_calls']),phase_seconds=trace['phase_seconds'],
        decode_counters=trace['decode_counters'],total_profile_calls=stats.total_calls,
        total_profile_exclusive_wall_seconds=stats.total_tt,top_exclusive=exclusive[:80],
        top_cumulative=cumulative[:80],selected_functions=sorted(relevant,key=lambda r:r['cumulative_wall_seconds'],reverse=True),
        cumulative_times_overlap=True,C_wait_wall_included=True,pure_CPU_cycles_claimed=False,
        representative_full463_timing=False,device_cost_gain_proved=False,formal_Goodput=False)
    with OUT.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(profile_seconds=result['actual_profiled_original_generate_seconds'],
        normal_target_calls=result['actual_normal_target_calls'],top_exclusive=exclusive[:20],
        selected_functions=result['selected_functions']),ensure_ascii=False))


if __name__=='__main__':main()
