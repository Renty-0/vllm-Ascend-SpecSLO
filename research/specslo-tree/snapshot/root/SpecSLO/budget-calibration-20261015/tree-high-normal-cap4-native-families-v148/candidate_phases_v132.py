"""Exactly one candidate warm65/zero65/zero65/seal/full463, one model."""
from pathlib import Path
from window_cost_state_v132 import snapshot as window_cost_snapshot
from combined_controls_v113 import proposal_counter_snapshot
UPDATE_FIELDS=('aclgraph_task_update_skipped_replays','aclgraph_draft_task_update_replays',
 'aclgraph_draft_task_update_tasks','aclgraph_target_task_update_replays','aclgraph_target_task_update_tasks')

def update_snapshot(engine):
    metrics=engine.graph_metrics();out={}
    for field in UPDATE_FIELDS:
        value=metrics[field]
        if type(value) is not int or value<0:raise RuntimeError('V132 actual task-update counter invalid: '+field)
        out[field]=value
    return out

def split_snapshot(engine,*,reset_normal_admission=False):
    from poisson_coverage_probe import _policy_snapshot
    _policy_snapshot(engine)
    fn=engine.graph_metrics.__func__;cells=dict(zip(fn.__code__.co_freevars,fn.__closure__ or ()))
    policy=cells['policy'].cell_contents
    from normal_admission_v140 import reset,snapshot as admission_snapshot
    from gain_state_v140 import snapshot as gain_snapshot
    generation=getattr(engine,'_dynamic_tree_generation',0)
    if reset_normal_admission:reset(policy,generation+1)
    keys=('normal_cap4_shape_calls','normal_cap5_shape_calls','normal_cap4_limited_owners',
        'partial_eager_grant_deferred_B1','partial_eager_grant_deferred_B2',
        'eager_deferred_shared_steps4_rows','eager_deferred_shared_steps4_candidates',
        'split_draft_deferred_reserved_candidates','split_draft_final_normal_candidates',
        'split_draft_final_eager_candidates','split_draft_final_unused_candidates',
        'actual_draft_nonempty_work_calls','actual_draft_work_rows','actual_draft_normal_rows','actual_draft_eager_rows',
        *(f'draft_normal_N{n}_shape_calls' for n in range(33)),
        *(f'actual_draft_normal_N{n}_work_calls' for n in range(33)))
    values={key:policy.counts[key] for key in keys}
    fast=dict(getattr(engine,'_dynamic_tree_draft_counts',{}))
    if any(type(value) is not int or value<0 for value in (*values.values(),*fast.values())):raise RuntimeError('V132 actual HOST policy/FastV5 Counter field invalid')
    return dict(policy=values,FastV5=fast,normal_admission=admission_snapshot(policy),
        gain_state=gain_snapshot(policy,engine_generation=generation),
        Counter_missing_zero_semantics=True,missing_Native_fields_defaulted=False)

def split_delta(before,after):
    return {family:{key:after[family].get(key,0)-before[family].get(key,0)
        for key in before[family].keys()|after[family].keys()} for family in ('policy','FastV5')}

def schedule():return (('warmup65',65),('zero65-1',65),('zero65-2',65),('seal',0),('full463',463))

def run_candidate(engine,*,controls,scope,stage,run_trace,root,checkpoint=None):
    traces=[];states=[]
    stage('candidate/install-controls',lambda:controls.select('candidate',engine=engine))
    directory=Path(root)/'candidate';directory.mkdir(exist_ok=True)
    for role,count in schedule():
        if role=='seal':
            def seal():
                engine.seal_graph_cache()
                if not engine.graph_runner.graph_cache_sealed:raise RuntimeError('V132 original Native seal missing')
            stage('candidate/unified-seal',seal);continue
        before=stage('candidate/'+role+'/controls-before',controls.observe)
        counters=stage('candidate/'+role+'/proposal-counts-before',lambda:proposal_counter_snapshot(engine))
        updates=stage('candidate/'+role+'/update-counts-before',lambda:update_snapshot(engine))
        split=stage('candidate/'+role+'/split-work-counts-before',lambda:split_snapshot(engine,reset_normal_admission=True))
        window_state=stage('candidate/'+role+'/window-state-before',lambda:window_cost_snapshot(engine))
        trace=run_trace(engine,count=count,role=role,directory=directory,qualified_s35_scope=scope)
        window_state_after=stage('candidate/'+role+'/window-state-after',lambda:window_cost_snapshot(engine))
        trace['actual_window_cost_state']=dict(before=window_state,after=window_state_after)
        after_counts=stage('candidate/'+role+'/proposal-counts-after',lambda:proposal_counter_snapshot(engine))
        after_updates=stage('candidate/'+role+'/update-counts-after',lambda:update_snapshot(engine))
        after_split=stage('candidate/'+role+'/split-work-counts-after',lambda:split_snapshot(engine))
        trace['actual_normal_admission']=dict(before=split['normal_admission'],after=after_split['normal_admission'])
        trace['actual_gain_state']=dict(before=split['gain_state'],after=after_split['gain_state'])
        trace['actual_proposal_counts']=dict(before=counters,after=after_counts,delta={k:after_counts[k]-counters[k] for k in counters})
        counter_fields=('policy','FastV5','Counter_missing_zero_semantics','missing_Native_fields_defaulted')
        trace['actual_split_work_counts']=dict(before={k:split[k] for k in counter_fields},
            after={k:after_split[k] for k in counter_fields},delta=split_delta(split,after_split))
        trace['actual_task_update_counts']=dict(before=updates,after=after_updates,delta={k:after_updates[k]-updates[k] for k in updates})
        after=stage('candidate/'+role+'/controls-after',controls.observe)
        states.append(dict(role=role,before=before,after=after));traces.append(trace)
        if checkpoint is not None:stage('candidate/'+role+'/checkpoint',lambda:checkpoint(traces,states))
    stage('candidate/restore-original-controls',lambda:controls.select('original',engine=engine))
    return traces,states
