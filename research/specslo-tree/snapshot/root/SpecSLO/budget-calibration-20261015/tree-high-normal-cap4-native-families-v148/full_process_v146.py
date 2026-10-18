"""Untimed own-process qualification bridge. No Native flags are assigned."""
from contextlib import contextmanager
from scope_v146 import require,cases,target_key,inventory_record,validate_full_configuration

FAMILIES=('entries','draft_entries','target_entries')


def capture_inventory(engine):
    runner=engine.graph_runner
    expected=dict(entries=0,draft_entries=60,target_entries=54) if engine.is_draft else dict(entries=0,draft_entries=0,target_entries=247)
    require({name:len(getattr(runner,name)) for name in FAMILIES}==expected,
        'own component starts strictly after original 86+28 or 201+44+S19/S35 gates')
    if not engine.is_draft:
        for case in cases():
            require(target_key(case,hidden=True) not in runner.target_entries,
                'two diagnostic hidden keys must genuinely be new to this full process')
            require(target_key(case) in runner.target_entries,
                'both production greedy keys must already belong to the original44 inventory')
    return dict(entries={name:dict(getattr(runner,name)) for name in FAMILIES},
        states={name:{key:(entry.runtime_validated,getattr(entry,'validated_real_row_count',None))
            for key,entry in getattr(runner,name).items()} for name in FAMILIES},
        disabled=set(runner.disabled_entry_keys),record=inventory_record(engine))


def validate_inventory_delta(before,after,*,is_draft):
    require(set(before)==set(after)==set(FAMILIES),'complete Native inventory families')
    for family in FAMILIES:
        wanted={repr(target_key(case,hidden=True)) for case in cases()} if not is_draft and family=='target_entries' else set()
        require(before[family]==sorted(set(before[family])) and after[family]==sorted(set(after[family]))
            and set(before[family])<=set(after[family]) and set(after[family])-set(before[family])==wanted,
            'full process adds only two causal hidden keys, no draft/greedy/generic key')


def finish_inventory(engine,before):
    runner=engine.graph_runner;after=inventory_record(engine)
    validate_inventory_delta(before['record'],after,is_draft=engine.is_draft)
    require(set(runner.disabled_entry_keys)==before['disabled'],'qualification must not disable a Native key')
    old_states={};natural_changes={}
    for name in FAMILIES:
        current=getattr(runner,name);old_states[name]={};natural_changes[name]={}
        for key,entry in before['entries'][name].items():
            require(current[key] is entry,'every inherited Native entry object remains authoritative')
            old=before['states'][name][key]
            now=(entry.runtime_validated,getattr(entry,'validated_real_row_count',None))
            require(not old[0] or now[0] is True,'inherited Native validation cannot regress')
            require(old[1] is None or (now[1] is not None and now[1]>=old[1]),'validated real-row bound cannot regress')
            old_states[name][repr(key)]=list(old)
            if old!=now:natural_changes[name][repr(key)]=dict(before=list(old),after=list(now))
    if not engine.is_draft:
        for case in cases():
            entry=runner.target_entries[target_key(case,hidden=True)]
            require(entry.runtime_validated is True,'both new full-process hidden gates close naturally')
    return dict(before=before['record'],after=after,inherited_entry_identity_preserved=True,
        inherited_states_before=old_states,natural_validation_changes=natural_changes,
        Native_flags_assigned=False,disabled_keys_unchanged=True,new_hidden_keys=0 if engine.is_draft else 2,
        new_draft_keys=0,new_greedy_keys=0,new_generic_keys=0)


@contextmanager
def track_actual_target_key(engine):
    """Observe original per-instance execution provenance, restoring ownership."""
    runner=engine.graph_runner;name='_record_execution'
    had=name in runner.__dict__;previous=runner.__dict__.get(name);original=getattr(runner,name)
    key_had='_v146_last_target_key' in runner.__dict__;key_before=runner.__dict__.get('_v146_last_target_key')
    def tracked(kind,execution,output,*,entry_key=None,token_rows=()):
        if kind=='target':runner._v146_last_target_key=entry_key
        return original(kind,execution,output,entry_key=entry_key,token_rows=token_rows)
    setattr(runner,name,tracked)
    try:yield
    finally:
        if had:setattr(runner,name,previous)
        else:runner.__dict__.pop(name,None)
        if key_had:runner._v146_last_target_key=key_before
        else:runner.__dict__.pop('_v146_last_target_key',None)


def capture_fixture_state(engine):
    names=('_defer_tree_materialization','_defer_normal_tree_materialization')
    return dict(engine={name:(name in engine.__dict__,engine.__dict__.get(name)) for name in names},
        runner_top2=('_dynamic_tree_top2' in engine.graph_runner.__dict__,
            engine.graph_runner.__dict__.get('_dynamic_tree_top2')),
        source_generation=getattr(engine,'_dynamic_tree_generation',0))


def finish_fixture_state(engine,state):
    require(engine.cache_allocation is None and engine.cache_block_tables is None,
        'original release completes before clearing a fixture-only ledger')
    require(getattr(engine,'_dynamic_tree_generation',0)==state['source_generation'],
        'qualification must not impersonate a production fresh-decode generation')
    for name,(had,value) in state['engine'].items():
        if had:setattr(engine,name,value)
        else:engine.__dict__.pop(name,None)
    had,value=state['runner_top2']
    if had:engine.graph_runner._dynamic_tree_top2=value
    else:engine.graph_runner.__dict__.pop('_dynamic_tree_top2',None)
    engine._dynamic_tree_known_KV={}
    engine._dynamic_tree_fast_profile_calls=0
    require(not getattr(engine,'_linear_draft_fia_persistent_staging_pool',{}),
        'original release clears allocation-owned FIA staging')
    return dict(original_cache_release_completed=True,known_KV_ledger_empty=True,
        original_allocate_will_clear_ledger_and_profile_again=True,defer_flags_restored=True,
        runner_top2_restored=True,source_generation_unchanged=state['source_generation'],
        cumulative_FastV5_counts_preserved_and_phase_deltas_required=True,
        Native_entries_flags_and_graph_owned_buffers_cleared=False,
        diagnostic_tail_buffers_not_consumed_as_production_values=True)


def reset_fixture_gain(policy):
    """Fresh untimed fixture/production learning boundary; calibrated prior stays."""
    from copy import deepcopy
    prior=deepcopy(policy.base);previous=deepcopy(policy.observed)
    policy.observed.clear()
    require(policy.base==prior and not policy.observed,'only observed fixture gain is cleared')
    return dict(base=prior,cleared_observed=[dict(slo_class=c,B=b,rounds=v[0],delivered=v[1])
        for (c,b),v in sorted(previous.items())],observed_after=[],calibrated_prior_unchanged=True)


def run_full_process_component(engine,native,compiled,*,directory,policy):
    from pathlib import Path
    from actor_v146 import run_actor,save,stage4
    from audit_v146 import audit_reports
    events=[];directory=Path(directory)/'variable-B4-own-full-process';directory.mkdir(exist_ok=True)
    stage=lambda name,fn:stage4(engine,events,'V146/'+name,fn)
    stage('full-configuration',lambda:validate_full_configuration(engine))
    gain_before=stage('fresh-fixture-gain',lambda:reset_fixture_gain(policy))
    fixture_state=stage('capture-fixture-only-state',lambda:capture_fixture_state(engine))
    before=stage('preserve-inherited-Native-entries',lambda:capture_inventory(engine))
    try:
        with track_actual_target_key(engine):
            report=run_actor(engine,native,compiled,directory,events)
    except BaseException:
        # Only an exception-side local checkpoint; never masks the original error.
        try:
            save(directory,f'rank{engine.rank}-partial-component-error.json.gz',
                dict(partial=getattr(engine,'_v146_partial',None),formal_Goodput=False,
                    full527_qualification=False,retained_boundary_prefix=len(events)))
        except BaseException:pass
        raise
    report['inventory_delta']=stage('only-two-new-hidden-keys',lambda:finish_inventory(engine,before))
    report['gain_reset']=dict(before=gain_before,after=stage('fresh-production-gain',lambda:reset_fixture_gain(policy)))
    report['production_fresh_boundary']=stage('discard-fixture-ledger-and-restore-defer-flags',
        lambda:finish_fixture_state(engine,fixture_state))
    stage('qualification-cache-released-before-production',lambda:require(engine.cache_allocation is None
        and engine.cache_block_tables is None and not engine.graph_runner.graph_cache_sealed,
        'local fixture releases both banks and leaves unified production seal pending'))
    report['boundary_events']=events
    report['retained_boundary_prefix']=dict(stage_count=len(events),
        last_label=events[-1]['label'] if events else None,
        saving_stage_and_later_audit_not_retained=True)
    stage('retain-own-full-process-component',lambda:save(directory,f'rank{engine.rank}-component.json.gz',report))
    # All4 only after every rank retained its report; audit reads canonical JSON.
    proof=stage('audit-retained-own-full-process-component',lambda:audit_reports(directory))
    require(proof['component_numerical_qualified'] is True,'own full-process natural missing/independent protocol qualification incomplete')
    return dict(directory=str(directory),audit=proof,Native_flags_assigned=False,component_sealed=False,
        full527_own_process_actual_component_qualified=True,production_two65zero_still_required=True,
        production_full463_qualified=False)
