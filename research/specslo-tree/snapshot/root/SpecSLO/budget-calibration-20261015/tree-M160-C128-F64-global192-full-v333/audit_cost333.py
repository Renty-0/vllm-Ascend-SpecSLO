"""Independently recompute current-model target cost from retained samples."""
import hashlib
import json
import math
from pathlib import Path
import statistics


def audit(directory,proofs,*,expected_source_sha256):
    directory=Path(directory).resolve();rows=[];evidence=[];profiles=set();source_pins=set()
    expected=[]
    for n in(4,8,12,16,20,24,28,31):
        for forks in sorted({1,min(n,32-n,(128-4*n)//4,12)}):expected.append((n,forks))
    expected.append((32,0))
    keys=('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays',
          'aclgraph_failed_captures','aclgraph_capacity_fallbacks','aclgraph_shape_fallbacks')
    for rank in(1,2,3):
        path=directory/'paired-current-target-cost333'/f'rank{rank}-target-cost.json'
        raw=path.read_bytes();item=json.loads(raw);profile=item['profile_sha256'];profiles.add(profile)
        source_pins.add(item['source_freeze_sha256'])
        assert item['rank']==rank and item['run_directory']==str(directory)
        assert item['model_dtype']=='torch.bfloat16' and item['fresh_math_cases']==532
        assert item['all_math_bitwise']is True and item['physical_FIA_rows']==160
        assert item['physical_FIA_groups']==32 and item['actual_projection_rows']==165
        assert item['chained_flush_334']is True and item['deferred_read_done']is False
        report=item['costs'];assert report['rank']==rank and report['target_math_on_rank']is True
        assert report['no_new_capture_or_validation_in_costs']is True
        assert all(report['graph_counters_before'][k]==report['graph_counters_after'][k]for k in keys)
        assert [(r['N'],r['branch_owners'])for r in report['rows']]==expected
        for row in report['rows']:
            names=tuple(range(4,9))if row['branch_owners']else(4,)
            assert row['rounds']==len(row['pairs'])==20 and row['original_greedy_output_stable']is True
            samples={b:[]for b in names}
            for i,pair in enumerate(row['pairs']):
                assert tuple(pair['order'])==names[i%len(names):]+names[:i%len(names)]
                assert set(pair['milliseconds'])=={str(b)for b in names}
                for b in names:
                    elapsed=pair['milliseconds'][str(b)]
                    assert type(elapsed)in(float,int)and math.isfinite(elapsed)and elapsed>0
                    samples[b].append(elapsed)
            rows.append(dict(rank=rank,N=row['N'],branch_owners=row['branch_owners'],
                             median_ms={str(b):statistics.median(v)for b,v in samples.items()}))
        evidence.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
    assert len(profiles)==1 and source_pins=={expected_source_sha256} and len(rows)==48
    forks=[r for r in rows if r['branch_owners']]
    single={(r['rank'],r['N']):r for r in forks if r['branch_owners']==1}
    setup=max([0.]+[r['median_ms'][str(b)]/r['median_ms']['4']-1 for r in single.values()for b in range(5,9)])
    slopes=[0.]
    for r in forks:
        if r['branch_owners']>1:
            first=single[r['rank'],r['N']]
            for b in range(5,9):
                slopes.append((r['median_ms'][str(b)]/r['median_ms']['4']-first['median_ms'][str(b)]/first['median_ms']['4'])/(r['branch_owners']-1))
    slope=max(slopes)
    for rank,proof in enumerate(proofs):
        cal=proof['paired_target_cost_calibration']
        assert cal['setup_ratio']==setup and cal['per_additional_branch_ratio']==slope
        assert cal['source_freeze_sha256']==next(iter(source_pins)) and cal['profile_sha256']==next(iter(profiles))
        assert cal['evidence']==evidence and cal['target_costs']==rows
        assert cal['same_instance_before_performance']is True and cal['real_draft_cost_measured']is False
        assert proof['actual_cold_math_graph_calls_only']is True
        assert proof['untimed_cost_graph_call_count']==(0 if rank==0 else 1596)
        assert proof['all_prefixes_exact_after_cost']is(None if rank==0 else True)
    return dict(source_sha256=next(iter(source_pins)),profile_sha256=next(iter(profiles)),
        setup_ratio=setup,per_additional_branch_ratio=slope,panels_per_target=16,rotations=20,
        all48_target_cost_rows_recomputed=True,raw_samples_and_graph_counter_arithmetic_verified=True,
        original_cold532_and_untimed1596_graph_call_boundaries_separate=True,
        real_draft_cost_measured=False,whole_workload_speedup_inferred=False)
