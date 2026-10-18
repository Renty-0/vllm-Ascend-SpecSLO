"""Observe the original chained-flush contract before costly prefill capture.

This gate reads native model/route state. It never enables routes, changes a
decision, sets native qualification flags, or requests deferred read-done.
"""
import hashlib
import json
from pathlib import Path
import traceback

from finite_inventory_plan import MC2_ROWS

P=Path(__file__).resolve().parent
PROFILE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/mc2-profile.json')
PROFILE_SHA256='ad9e0648810ed697679ee84b8bbc2939ea14bfb687d0215d27f403ad5526db1c'
OPERATOR='tp3_native_matmul+allreduce_add_rmsnorm'


def evaluate(engine, profile, disk_sha256):
    """Pure host inspection; synthetic CPU fixtures are not NPU qualification."""
    from vllm_ascend.spec_decode.pearl.mc2 import MC2Profile, MC2StaticRouteManifest, _mc2_profile_sha256
    checks={}
    def check(name,value):checks[name]=bool(value)
    expected=set(MC2_ROWS)
    canonical=_mc2_profile_sha256(profile)
    record=dict(rank=int(engine.rank),role='draft' if engine.is_draft else 'target',
        phase='before_exact_prefill',profile_path=str(PROFILE),profile_disk_sha256=disk_sha256,
        expected_profile_disk_sha256=PROFILE_SHA256,profile_canonical_sha256=canonical,
        profile_operator=profile.metadata.get('operator'),
        profile_standalone_chained_flush=profile.metadata.get('standalone_chained_flush'),
        profile_deferred_read_done_chain_present='deferred_read_done_chain' in profile.metadata,
        planned_full_chain_rows=sorted(expected),checks=checks,
        native_qualification_flags_modified=False)
    check('frozen_profile_disk_hash',disk_sha256==PROFILE_SHA256)
    check('profile_operator',profile.metadata.get('operator')==OPERATOR)
    check('profile_standalone_chained_flush',profile.metadata.get('standalone_chained_flush') is True)
    check('profile_target_TP3',profile.metadata.get('tensor_parallel_size')==3)
    check('original_MC2_enabled',engine.config.enable_mc2 is True)
    if not engine.is_draft:
        model=engine.model
        actual_profile=getattr(model.config,'pearl_mc2_profile',None)
        manifest=getattr(engine,'mc2_static_route_manifest',None)
        actual_rows=set(getattr(model,'_mc2_chained_row_counts',()))
        deferred=set(getattr(model,'_mc2_deferred_row_counts',()))
        routes=tuple(getattr(manifest,'routes',()))
        record.update(manifest_type=type(manifest).__name__,manifest_route_count=len(routes),
            manifest_profile_sha256=getattr(manifest,'profile_sha256',None),
            manifest_digest=getattr(manifest,'digest',None),
            model_chained_row_counts=sorted(actual_rows),
            model_deferred_row_counts_observed_only=sorted(deferred),
            missing_chained_rows=sorted(expected-actual_rows),
            unexpected_chained_rows=sorted(actual_rows-expected),route_details=[])
        check('model_profile_is_native_normalized_profile',isinstance(actual_profile,MC2Profile))
        actual_canonical=(_mc2_profile_sha256(actual_profile) if isinstance(actual_profile,MC2Profile) else None)
        record['model_profile_canonical_sha256']=actual_canonical
        check('model_profile_matches_frozen_profile',actual_canonical==canonical)
        check('manifest_is_original_native_type',isinstance(manifest,MC2StaticRouteManifest))
        check('manifest_profile_matches_frozen_profile',getattr(manifest,'profile_sha256',None)==canonical)
        check('model_routes_frozen',getattr(model,'_mc2_static_routes_frozen',False) is True)
        check('engine_and_model_share_original_manifest',getattr(model,'_mc2_static_route_manifest',None) is manifest)
        check('original64layers',len(model.layers)==64)
        check('original128routes',len(routes)==128)
        check('ordinary_chained_rows_equal_planned22',actual_rows==expected)
        slots=[];disabled=[];bad_decisions=[]
        for route in routes:
            kind=getattr(route,'projection_kind',None);layer=getattr(route,'layer_index',None)
            slots.append((kind,layer))
            enabled=getattr(route,'enabled',False) is True
            decisions=getattr(route,'decisions',{})
            missing=[];unqualified=[]
            for rows in sorted(expected):
                decision=decisions.get(rows)
                if decision is None:missing.append(rows)
                elif getattr(decision,'qualified',False) is not True:
                    unqualified.append(dict(rows=rows,reason=str(getattr(decision,'reason',''))))
            detail=dict(projection_kind=kind,layer_index=layer,enabled=enabled,
                disabled_reason=str(getattr(route,'disabled_reason','')),
                operator_kind=getattr(route,'operator_kind',None),
                planned_decisions_present=len(expected)-len(missing),missing_rows=missing,
                unqualified_rows=unqualified)
            record['route_details'].append(detail)
            if not enabled:disabled.append(detail)
            if missing or unqualified:bad_decisions.append(detail)
        record['disabled_routes']=disabled;record['routes_with_missing_or_unqualified_rows']=bad_decisions
        check('complete_attention_down_route_identity',len(slots)==128 and set(slots)=={
            (kind,layer) for kind in ('attention','down') for layer in range(64)})
        check('all_original_routes_enabled',not disabled and len(routes)==128)
        check('all_original_route_operators',len(routes)==128 and all(
            getattr(route,'operator_kind',None)==OPERATOR for route in routes))
        check('every_planned_row_qualified_on_every_route',not bad_decisions and len(routes)==128)
    record['failed_checks']=[name for name,passed in checks.items() if not passed]
    record['local_passed']=not record['failed_checks']
    return record


def preflight(engine):
    """All four ranks finish a CPU vote before any exact-prefill capture."""
    import torch
    import torch.distributed as dist
    from vllm_ascend.spec_decode.pearl.mc2 import normalize_mc2_profile
    try:
        payload=PROFILE.read_bytes()
        profile=normalize_mc2_profile(json.loads(payload))
        record=evaluate(engine,profile,hashlib.sha256(payload).hexdigest())
    except BaseException:
        record=dict(rank=int(engine.rank),role='draft' if engine.is_draft else 'target',
            phase='before_exact_prefill',local_passed=False,failed_checks=['inspection_exception'],
            inspection_traceback=traceback.format_exc(),native_qualification_flags_modified=False)
    # This is the same all-world Gloo group already required by the original
    # MC2 route admission. Target-only or verification subgroup votes cannot
    # leave a draft worker waiting after a target-side rejection.
    group=engine.groups.world_coordination_group
    if group is None or dist.get_backend(group)!='gloo' or dist.get_world_size(group)!=4:
        raise RuntimeError('Original all4rank world Gloo group is required for MC2 preflight')
    votes=[None]*4
    dist.all_gather_object(votes,dict(rank=int(engine.rank),local_passed=record['local_passed'],
        failed_checks=record['failed_checks']),group=group)
    passed=({row['rank'] for row in votes}==set(range(4))
            and all(row['local_passed'] for row in votes))
    record.update(all4rank_votes=votes,all4rank_contract_passed=passed,
        status='passed' if passed else 'failed')
    write_error=None
    try:
        directory=P/'startup-progress';directory.mkdir(exist_ok=True)
        path=directory/f'MC2-preflight-rank{engine.rank}.json'
        path.write_text(json.dumps(record,indent=2)+'\n')
    except BaseException:write_error=traceback.format_exc()
    failure=torch.tensor([int(not passed or write_error is not None)],dtype=torch.int32)
    dist.all_reduce(failure,op=dist.ReduceOp.MAX,group=group)
    if int(failure.item()):
        raise RuntimeError('Original MC2 chained-flush preflight rejected before prefill; '
            f'all4rank votes={votes!r}; diagnostic write error={write_error!r}')
    return record


def audit(directory):
    """Require complete successful device-startup observations on every rank."""
    records=[]
    for rank in range(4):
        path=Path(directory)/f'MC2-preflight-rank{rank}.json'
        row=json.loads(path.read_text())
        if (row.get('rank')!=rank or row.get('status')!='passed'
                or row.get('phase')!='before_exact_prefill'
                or not row.get('local_passed') or not row.get('all4rank_contract_passed')
                or row.get('failed_checks') or not row.get('checks')
                or not all(row['checks'].values())
                or row.get('native_qualification_flags_modified') is not False
                or {vote['rank'] for vote in row.get('all4rank_votes',[])}!=set(range(4))
                or not all(vote['local_passed'] for vote in row['all4rank_votes'])):
            raise RuntimeError('MC2 preflight device receipt is absent/incomplete/rejected: '+str(path))
        records.append(row)
    return dict(passed=True,rank_evidence=[dict(rank=row['rank'],role=row['role'],
        manifest_digest=row.get('manifest_digest'),route_count=row.get('manifest_route_count'),
        chained_rows=row.get('model_chained_row_counts')) for row in records])
