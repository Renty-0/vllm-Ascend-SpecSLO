"""Failure-only retention of existing HOST values; never a successful trace.

No tensor reads/ops, clocks, collectives, model calls or success-path observer.
Native fatal cleanup and wrapper restoration remain owned by the original code.
"""
import gzip
import hashlib
import json
import math
from pathlib import Path
import traceback

MAX_RAW_BYTES=64*1024**2
MAX_GZIP_BYTES=8*1024**2
MAX_HOST_ITEMS=4000000
MAX_REQUESTS=463


def _error(error):
    # Do not format an arbitrary object (notably a tensor) as diagnostic data.
    return dict(type=type(error).__module__+'.'+type(error).__qualname__,message=str(error)[:4096])


def _host(value,*,_depth=0,_budget=None):
    if _budget is None:_budget=[MAX_HOST_ITEMS]
    _budget[0]-=1
    if _budget[0]<0 or _depth>32:raise ValueError('HOST diagnostic item/depth limit')
    if value is None or type(value) in (str,bool,int):return value
    if type(value) is float:
        if not math.isfinite(value):raise ValueError('nonfinite HOST diagnostic scalar')
        return value
    if type(value) in (tuple,list):return [_host(v,_depth=_depth+1,_budget=_budget) for v in value]
    if isinstance(value,dict):
        if any(type(k) not in (str,int) for k in value):raise TypeError('unsupported HOST mapping key type')
        keys=[str(k) for k in value]
        if len(set(keys))!=len(keys):raise ValueError('HOST int/string mapping-key collision')
        return {str(k):_host(v,_depth=_depth+1,_budget=_budget) for k,v in value.items()}
    raise TypeError('non-HOST diagnostic value '+type(value).__module__+'.'+type(value).__qualname__)


def _ids(values,*,maximum=MAX_REQUESTS):
    if type(values) not in (tuple,list) or len(values)>maximum or any(type(i) is not int or i<0 for i in values):
        raise ValueError('bounded exact HOST integer IDs required')
    return list(values)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _identity(engine):
    generation=getattr(engine,'_dynamic_tree_generation',None);rank=getattr(engine,'rank',None)
    if type(generation) is not int or generation<=0 or type(rank) is not int or rank not in range(4):
        raise ValueError('actual rank/current source generation unavailable')
    return rank,generation


def _index(engine):
    rank,generation=_identity(engine);index=getattr(engine,'_v140_failure_evidence',None)
    if type(index) is not dict or index.get('rank')!=rank or index.get('source_generation')!=generation:
        index=dict(schema='V140.failure-evidence-index.v1',rank=rank,source_generation=generation,
            artifacts=[],retention_errors=[],failure_only=True,formal_Goodput=False,full463_completed=False)
        engine._v140_failure_evidence=index
    return index


def failure_index(engine):
    """Plain HOST index, with exact current-generation ownership or missing."""
    try:
        rank,generation=_identity(engine);value=getattr(engine,'_v140_failure_evidence',None)
        if type(value) is dict and value.get('rank')==rank and value.get('source_generation')==generation:
            return _host(value)
        return dict(rank=rank,source_generation=generation,artifacts=[],retention_errors=[],
            missing='no failure snapshot for this source generation',failure_only=True,formal_Goodput=False)
    except BaseException as error:
        return dict(artifacts=[],retention_errors=[_error(error)],missing='actual generation unavailable',failure_only=True,formal_Goodput=False)


def _section(report,name,builder):
    try:report[name]=_host(builder())
    except BaseException as error:report.setdefault('missing_sections',{})[name]=_error(error)


def _write(engine,directory,kind,value):
    rank,generation=_identity(engine);path=Path(directory)/f'rank{rank}-generation{generation}-{kind}.json.gz'
    raw=json.dumps(_host(value),separators=(',',':'),allow_nan=False).encode()
    if len(raw)>MAX_RAW_BYTES:raise ValueError('failure diagnostic uncompressed cap exceeded')
    data=gzip.compress(raw,mtime=0)
    if len(data)>MAX_GZIP_BYTES:raise ValueError('failure diagnostic compressed cap exceeded')
    # Directory belongs to the actual attempt. Never create a fallback path,
    # overwrite another artifact or update a previous failure receipt.
    if not path.parent.is_dir():raise ValueError('actual private failure directory does not exist')
    with path.open('xb') as stream:stream.write(data)
    info=dict(kind=kind,path=str(path.resolve()),bytes=len(data),uncompressed_bytes=len(raw),sha256=hashlib.sha256(data).hexdigest(),
        rank=rank,source_generation=generation)
    _index(engine)['artifacts'].append(info)
    return info


def _states(values):
    if type(values) not in (tuple,list) or len(values)>MAX_REQUESTS:raise ValueError('actual bounded logical state sequence required')
    result=[]
    for index,state in enumerate(values):
        tokens=getattr(state,'token_ids',None);prompt=getattr(state,'prompt_length',None);committed=getattr(state,'committed_length',None)
        if (type(tokens) is not list or len(tokens)>4096 or any(type(v) is not int for v in tokens)
                or type(prompt) is not int or type(committed) is not int or not 0<=prompt<=committed<=len(tokens)):
            raise ValueError('actual HOST token/prompt/committed lengths inconsistent')
        result.append(dict(request_index=index,prompt_length=prompt,committed_length=committed,total_token_ids=len(tokens),
            token_ids_sha256=_digest(tokens),committed_completion_token_ids=list(tokens[prompt:committed]),
            uncommitted_suffix_token_ids=list(tokens[committed:]),aborted=getattr(state,'aborted',False)))
    return result


def _runtime_states(values):
    if not isinstance(values,dict) or len(values)>MAX_REQUESTS:raise ValueError('actual runtime state registry required')
    fields=('request_index','home_batch_id','slo_tpot_ms','slo_class','max_gamma','output_token_budget',
        'delivered_tokens','decode_elapsed_ms','decode_start_elapsed_ms','arrival_wait_ms','acceptance_ema','verification_rounds')
    result=[]
    for key,state in values.items():
        if type(key) is not int or key<0:raise ValueError('actual runtime state request index required')
        result.append(dict(registry_index=key,**{name:getattr(state,name,None) for name in fields}))
    return result


def _plan(value):
    if value is None:return None
    fields=('plan_id','target_home_batch_id','draft_home_batch_id','target_request_indices','normal_draft_request_indices',
        'eager_candidate_indices','target_candidate_budgets','verification_candidate_budget','deferred_target_request_indices')
    result={name:getattr(value,name,None) for name in fields}
    phase=getattr(value,'phase',None);result['phase']=getattr(phase,'value',None)
    return result


def _grant_binding(frame,current_plan):
    budget=frame.get('budget_plan');budget_id=getattr(budget,'plan_id',None)
    plan_id=getattr(current_plan,'plan_id',None)
    controller=frame.get('controller');next_plan_id=getattr(controller,'plan_id',None)
    agree=type(plan_id) is int and type(budget_id) is int and plan_id==budget_id
    record=dict(budget_plan_exists=budget is not None,observed_budget_plan_id=budget_id,
        observed_execution_plan_id=plan_id,observed_controller_next_plan_id=next_plan_id,
        observed_round_count=frame.get('round_count'),local_plan_ids_agree=agree,
        current_round_binding_proved=False,current_round_grants_missing=True,
        reason='Local plan/budget references can both belong to an earlier round; ID equality alone is insufficient')
    if budget is not None:
        record['observed_possibly_stale_grants']=dict(normal_budgets=dict(budget.normal_budgets),
            eager_budgets=dict(budget.eager_budgets),deferred_normal_request_indices=list(budget.deferred_normal_request_indices))
    # No previous-cycle work_indices/work_plans or engine.last_worker_decode_*
    # are promoted to current. No new per-round witness is installed.
    return record


def retain_decode_failure(engine,error,frame):
    """Called ONLY inside the original decode exception, before fatal cleanup.

    This routine catches every retention error. Caller still guards the import
    itself and unconditionally executes original fatal_cleanup_all4 afterwards.
    """
    try:
        index=_index(engine);rank,generation=_identity(engine)
        if type(frame) is not dict or frame.get('self') is not engine:raise ValueError('actual failing decode locals must own this engine')
        report=dict(schema='V140.failed-decode-current-HOST-state.v1',rank=rank,source_generation=generation,
            error=_error(error),failure_only=True,current_counters_source='failing decode locals, before original fatal cleanup',
            local_loop_fields_current_round_binding_proved=False,
            local_loop_fields_may_precede_failure_round=True,
            engine_last_worker_decode_fields_used_as_current=False,returned_generate_results=None,
            full463_completed=False,formal_Goodput=False,canonical_JSON_type_projection=True,
            tensor_values_or_methods_read=False,new_clock_or_collective=False)
        for name in ('round_count','last_cycle_ms','cycle_started','projected_wait_ms','verification_roof','actual_candidates'):
            _section(report,name,lambda name=name:frame[name])
        for name in ('active','pending_admission','target_indices','normal','eager'):
            _section(report,name,lambda name=name:_ids(frame[name]))
        for name in ('counters','phase_seconds','decode_profile_seconds','decode_profile_detail_seconds','decode_timeline','host_timeline'):
            _section(report,name,lambda name=name:frame[name])
        current_plan=frame.get('plan')
        _section(report,'observed_possibly_stale_execution_plan',lambda:_plan(current_plan))
        _section(report,'budget_plan_binding',lambda:_grant_binding(frame,current_plan))
        _section(report,'runtime_states',lambda:_runtime_states(frame['states']))
        _section(report,'partial_local_states',lambda:_states(frame['local_states']))
        def target_payload_geometry():
            ids=_ids(frame['target_indices']);payloads=frame['target_payloads']
            if type(payloads) is not list or len(ids)!=len(payloads):raise ValueError('current target payload/owner count differs')
            result=[]
            for owner,payload in zip(ids,payloads,strict=True):
                plan=payload['plan'];state=frame['local_states'][owner]
                result.append(dict(request_index=owner,proposal_id=payload['ticket'].proposal_id,
                    B=plan.candidate_budget,prefix_len=plan.prefix_len,observed_state_root=state.token_ids[-1],current_round_binding_proved=False))
            return result
        _section(report,'observed_possibly_stale_target_payload_geometry',target_payload_geometry)
        _section(report,'completed_request_indices',lambda:sorted(frame['completed']))
        hooks=frame.get('_v75_hooks')
        if hooks is not None:_section(report,'pre_fatal_boundary_evidence',hooks.evidence)
        directory=getattr(engine,'_v101_output_directory',None)
        if not isinstance(directory,(str,Path)):raise ValueError('actual attempt output directory unavailable')
        return _write(engine,directory,'decode-failure-HOST-state',report)
    except BaseException as retained_error:
        try:_index(engine)['retention_errors'].append(dict(kind='decode-failure-HOST-state',error=_error(retained_error)))
        except BaseException:pass
        return None


def retain_trace_failure(engine,error,*,role,count,directory,source,prompts,params,stream,observation,
                         observations,begin_inputs,observation_errors,close_aborts,before,policy_before,
                         sealed_before,before_generation,native_current,policy_current):
    """Retain existing run_trace lists after its failed original all4 stage.

    No success trace/result or timing metric is fabricated. Native/policy
    callbacks are the already-imported HOST counter readers, not device queries.
    """
    try:
        index=_index(engine);rank,generation=_identity(engine)
        report=dict(schema='V140.failed-trace-existing-HOST-observations.v1',rank=rank,source_generation=generation,
            source_generation_before=before_generation,role=role,scope_count=count,error=_error(error),
            failure_only=True,full463_completed=False,formal_Goodput=False,generate_returned=False,
            returned_generate_results=None,elapsed_seconds=None,Goodput=None,
            engine_last_worker_decode_fields_used_as_current=False,
            decode_failure_index_before_trace=failure_index(engine),sealed_before=sealed_before,
            canonical_JSON_type_projection=True,tensor_values_or_methods_read=False,new_clock_or_collective=False)
        for name,value in [('source',source),('tokenized_inputs',prompts),('stream_events',stream),
                           ('normal_target_calls',observation.target_calls),('standalone_prefill_calls',observation.prefill_calls),
                           ('begin_inputs',begin_inputs),('close_abort_identities',close_aborts),
                           ('observation_errors',list(observation.errors)+list(observation_errors))]:
            _section(report,name,lambda value=value:value)
        _section(report,'sampling',lambda:[{name:getattr(p,name,None) for name in
            ('temperature','draft_temperature','max_tokens','ignore_eos','request_id','slo_class','slo_tpot_ms','arrival_ts')} for p in params])
        _section(report,'logical_events',lambda:[list(o.events) for o in observations])
        _section(report,'post_fatal_boundary_evidence',lambda:[o.hooks.evidence() for o in observations])
        _section(report,'native_before',lambda:before)
        _section(report,'native_after_failure',native_current)
        _section(report,'policy_before',lambda:policy_before)
        _section(report,'policy_after_failure',policy_current)
        if 'native_before' in report and 'native_after_failure' in report:
            _section(report,'native_delta_until_failure',lambda:{k:report['native_after_failure'][k]-report['native_before'][k] for k in report['native_before']})
        _section(report,'wrappers_restored_at_retention',lambda:observation.restored)
        return _write(engine,directory,'trace-failure-HOST-observations',report)
    except BaseException as retained_error:
        try:_index(engine)['retention_errors'].append(dict(kind='trace-failure-HOST-observations',error=_error(retained_error)))
        except BaseException:pass
        return None
