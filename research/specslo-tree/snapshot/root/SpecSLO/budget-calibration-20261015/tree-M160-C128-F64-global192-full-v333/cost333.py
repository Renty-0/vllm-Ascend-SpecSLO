"""Same-instance untimed target calibration; never imported latency evidence.

This ratio is a branch scheduling proxy. It excludes real draft generation
and does not guarantee lower latency or higher Goodput.
"""
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import statistics

_ISSUER=object()


def configurations():
    rows=[]
    for n in(4,8,12,16,20,24,28,31):
        for forks in sorted({1,min(n,32-n,(128-4*n)//4,12)}):rows.append((n,forks))
    return rows+[(32,0)]


def checked_costs(reports):
    if len(reports)!=3 or [report['rank']for report in reports]!=[1,2,3]:
        raise ValueError('All3 actual target cost roles required')
    rows=[]
    for report in reports:
        if report['target_math_on_rank']is not True or report['no_new_capture_or_validation_in_costs']is not True:
            raise ValueError('Actual captured target cost and stable inventory required')
        counters=('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays',
                  'aclgraph_failed_captures','aclgraph_capacity_fallbacks','aclgraph_shape_fallbacks')
        if any(k not in report['graph_counters_before'] or k not in report['graph_counters_after']
               or report['graph_counters_before'][k]!=report['graph_counters_after'][k] for k in counters):
            raise ValueError('Actual graph counter arithmetic changed during cost calibration')
        if [(row['N'],row['branch_owners'])for row in report['rows']]!=configurations():
            raise ValueError('All16 predeclared actual target cost configurations required')
        for row in report['rows']:
            names=tuple(range(4,9))if row['branch_owners']else(4,)
            if row['rounds']!=20 or len(row['pairs'])!=20 or row['original_greedy_output_stable']is not True:
                raise ValueError('Twenty synchronous rotated stable target pairs required')
            values={b:[]for b in names}
            for iteration,pair in enumerate(row['pairs']):
                if tuple(pair['order'])!=names[iteration%len(names):]+names[:iteration%len(names)]:
                    raise ValueError('Actual rotation order changed')
                if set(pair['milliseconds'])!={str(b)for b in names}:
                    raise ValueError('Complete budget timing observations required')
                for budget in names:
                    value=pair['milliseconds'][str(budget)]
                    if type(value)not in(float,int)or not math.isfinite(value)or value<=0:
                        raise ValueError('Finite positive actual elapsed time required')
                    values[budget].append(value)
            rows.append(dict(rank=report['rank'],N=row['N'],branch_owners=row['branch_owners'],
                             median_ms={str(b):statistics.median(v)for b,v in values.items()}))
    branches=[row for row in rows if row['branch_owners']>0]
    first={(row['rank'],row['N']):row for row in branches if row['branch_owners']==1}
    setup=max([0.]+[row['median_ms'][str(b)]/row['median_ms']['4']-1
                      for row in first.values()for b in range(5,9)])
    slopes=[0.]
    for row in branches:
        forks=row['branch_owners']
        if forks>1:
            initial=first[row['rank'],row['N']]
            for budget in range(5,9):
                slopes.append((row['median_ms'][str(budget)]/row['median_ms']['4']
                    -initial['median_ms'][str(budget)]/initial['median_ms']['4'])/(forks-1))
    slope=max(slopes)
    for row in branches:
        predicted=1+setup+slope*(row['branch_owners']-1)
        if any(predicted+1e-12<row['median_ms'][str(b)]/row['median_ms']['4']for b in range(5,9)):
            raise ValueError('Conservative envelope did not cover observed target ratios')
    return setup,slope,rows


@dataclass(frozen=True)
class Envelope:
    engine: object=field(repr=False,compare=False)
    policy: object=field(repr=False,compare=False)
    issuer: object=field(repr=False,compare=False)
    setup: float
    slope: float
    source_freeze_sha256: str
    profile_sha256: str
    evidence: tuple
    costs: tuple


def bind_current(policy,engine,directory,source_freeze_sha256):
    if hasattr(policy,'_ROOT333_cost_envelope'):
        raise RuntimeError('Same-instance cost calibration may be bound once')
    if not isinstance(source_freeze_sha256,str)or len(source_freeze_sha256)!=64:
        raise ValueError('Actual source freeze identity required')
    directory=Path(directory).resolve()
    if directory.parent!=Path(engine._v101_output_directory).resolve():
        raise ValueError('Cost records must belong to this current model run')
    reports=[];evidence=[];profiles=set()
    for rank in(1,2,3):
        path=directory/f'rank{rank}-target-cost.json'
        if path.is_symlink():raise ValueError('Current private cost evidence must be regular')
        raw=path.read_bytes();item=json.loads(raw)
        if(item['rank']!=rank or item['source_freeze_sha256']!=source_freeze_sha256
                or item['run_directory']!=str(directory.parent)
                or item['model_dtype']!='torch.bfloat16' or item['fresh_math_cases']!=532
                or item['all_math_bitwise']is not True
                or item['physical_FIA_rows']!=160 or item['physical_FIA_groups']!=32
                or item['actual_projection_rows']!=165
                or item['chained_flush_334']is not True or item['deferred_read_done']is not False):
            raise ValueError('Cost evidence escaped current fresh532 model/source/ABI scope')
        if rank==engine.rank and item['model_identity']!=id(engine.model):
            raise ValueError('Cost evidence model instance changed')
        reports.append(item['costs']);profiles.add(item['profile_sha256'])
        evidence.append((str(path),hashlib.sha256(raw).hexdigest()))
    if len(profiles)!=1:raise ValueError('Target cost profile identities disagree')
    if engine.rank and next(iter(profiles))!=engine.mc2_static_route_manifest.profile_sha256:
        raise ValueError('Current target MC2 profile changed during calibration')
    setup,slope,rows=checked_costs(reports)
    result=Envelope(engine,policy,_ISSUER,setup,slope,source_freeze_sha256,
                    next(iter(profiles)),tuple(evidence),tuple(rows))
    policy._ROOT333_cost_envelope=result
    return snapshot(policy)


def current(policy):
    result=getattr(policy,'_ROOT333_cost_envelope',None)
    if not isinstance(result,Envelope)or result.issuer is not _ISSUER or result.policy is not policy:
        raise RuntimeError('Fresh same-instance target cost calibration required before LIVE')
    return result


def ratio(policy,branches):
    if type(branches)is not int or not 0<=branches<=12:raise ValueError('Atmost12 branch owners')
    evidence=current(policy)
    return 1. if branches==0 else 1+evidence.setup+evidence.slope*(branches-1)


def snapshot(policy):
    evidence=current(policy)
    return dict(setup_ratio=evidence.setup,per_additional_branch_ratio=evidence.slope,
        source_freeze_sha256=evidence.source_freeze_sha256,profile_sha256=evidence.profile_sha256,
        evidence=[dict(path=path,sha256=sha)for path,sha in evidence.evidence],
        target_costs=list(evidence.costs),same_instance_before_performance=True,
        real_draft_cost_measured=False,latency_or_Goodput_guaranteed=False,
        actual_projection_rows=165,FIA_rows=160,FIA_groups=32,deferred_read_done=False)
