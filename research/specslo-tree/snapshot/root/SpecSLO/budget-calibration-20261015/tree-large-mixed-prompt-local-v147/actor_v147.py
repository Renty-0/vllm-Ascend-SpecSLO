"""Six real prompt/normal commit phases, two full local zeros, then own seal."""
from pathlib import Path
import gzip
import json
import time
import traceback
from scope_v147 import ARMS,NATIVE_COUNTERS,FAILURES,require,target_key,profile_scope,digest

def save(directory,name,value):
    raw=json.dumps(value,allow_nan=False,separators=(',',':')).encode()
    if name.endswith('.gz'):raw=gzip.compress(raw,mtime=0)
    with (Path(directory)/name).open('xb') as stream:stream.write(raw)

def native(engine):
    metrics=engine.graph_metrics();values={k:metrics[k] for k in NATIVE_COUNTERS}
    require(all(type(v) is int and v>=0 for v in values.values()),'all21 actual original Native counters')
    return values

def delta(before,after,*,zero=False):
    values={k:after[k]-before[k] for k in NATIVE_COUNTERS}
    require(all(v>=0 for v in values.values()) and not any(values[k] for k in FAILURES),
        'original Native failures/fallbacks/rejections absent')
    if zero:require(not any(values.values()),'complete fresh-history local zero discovers no graph or new validation')
    return dict(before=before,after=after,delta=values)

def stage4(engine,events,label,callback,*,share=False):
    import torch,torch.distributed as dist
    value=None;error=None;original=None;seq=len(events)
    try:value=callback();torch.npu.synchronize()
    except BaseException as exc:original=exc;error=traceback.format_exc()
    rows=[None]*4
    dist.all_gather_object(rows,dict(rank=engine.rank,sequence=seq,label=label,error=error,
        evidence=value if share and error is None else None),group=engine.groups.verification_coordination_group)
    events.append(dict(sequence=seq,label=label,ranks=rows))
    passed={r['rank'] for r in rows}==set(range(4)) and all(r['sequence']==seq and r['label']==label
        and r['error'] is None for r in rows)
    if not passed:
        if original is not None:raise original
        raise RuntimeError('V147 actual common stage failed '+repr(rows))
    engine._vote_spec_rhythm_prefill_finiteness()
    return rows if share else value

def inventory(engine):
    r=engine.graph_runner
    require(not r.entries,'no unrelated generic graphs or201inventory')
    if engine.is_draft:
        require(set(r.draft_entries)=={('draft-greedy:151936|steps:3|paged|confidence',32)},
            'only the actual shared3 M32 draft key')
        require(all(k[0]=='target-hidden:0|steps:1|paged' and 1<=k[1][0]<=32
            for k in r.target_entries),'only naturally encountered one-token-repair PA keys')
    else:
        require(set(r.target_entries)=={target_key(a) for a in ARMS} and not r.draft_entries,
            'exact M197/FULL128 and M645/FULL512 S35 hidden keys only')
    require(all(e.runtime_validated is True for e in (*r.target_entries.values(),*r.draft_entries.values()))
        and not r.disabled_entry_keys,'every actual Native key naturally changed/restored validated')
    return {k:sorted(map(repr,getattr(r,k))) for k in ('entries','draft_entries','target_entries')}

def run_actor(engine,native_module,compiled,directory,events):
    from bank_v147 import build_bank
    from draft_v147 import generate
    from target_v147 import qualify,reference,commit_reference,normal_cross_M_snapshot,normal_cross_M_compare
    from prompt_v147 import reserve,begin,prepare,commit,original_clock,publish,abort_partial
    from local_qualification_scope import SCOPES
    from original_tree_cycle_v75 import setup_normal_cycle,attach_normal_cycle_proposals,run_original_tree_cycle
    from vllm_ascend.spec_decode.pearl.mc2 import snapshot_mc2_dispatch_counters
    s=lambda label,callback:stage4(engine,events,label,callback)
    report=dict(schema='V147.large-mixed-local-worker.v1',rank=engine.rank,role='draft' if engine.is_draft else 'target',
        errors=[],traces=[],completed=False,passed_subset=False,MC2_645_qualified=False,adopted=False,
        cost_measured=False,formal_Goodput=False,full527_configuration_qualified=False,
        original_full_21scope26phase_and_three65_gates_still_required=True,
        profile_scope=s('original-profile-membership-not-MC2-proof',profile_scope))
    engine._v147_partial=report
    def construction():
        r=engine.graph_runner
        capacity=engine._v147_capacity_evidence
        require(capacity['original_max_graph_tokens']==512
            and capacity['effective_max_graph_tokens']==(512 if engine.is_draft else 645)
            and r.max_graph_tokens==capacity['effective_max_graph_tokens']
            and capacity['original_constructor_restored'] is True
            and capacity['ordinary_mixed_target_graph_enabled'] is False
            and capacity['native_qualification_flags_assigned'] is False,
            'actual local constructor capacity expansion with original class restored; never a graph qualification')
        require(engine.config.max_num_seqs==64 and engine.config.max_num_queued_seqs==66
            and engine._cache_storage_sequence_capacity==130 and tuple(engine._tree_graph_scratch_sequence_ids)==tuple(range(66,130))
            and tuple(engine._mixed_target_scratch_sequence_ids)==() and engine.prefix_cache.num_blocks==544,
            'actual local64/66/130 configuration and original544 pages')
        require(not r.entries and not r.target_entries and not r.draft_entries and not any(native(engine).values()),
            'constructor minimal; no201/44/86 inventory or foreign flags')
        return dict(maxseq=64,queued=66,storage=130,blocks=544,tree_scratch=list(range(66,130)),
            constructor_inventory_entries=0,layers=len(engine.model.layers),capacity=capacity)
    report['construction']=s('minimal-construction',construction)
    from mc2_inventory_contract import preflight
    report['original_MC2_static']=preflight(engine) # Original gate/rows unchanged; never append645.
    expected={};cross={};ordinal=0
    for pass_index in range(3):
        zero=pass_index>0
        for arm_index,arm in enumerate(ARMS):
            for scope in [v for v in SCOPES if v.arm_index==arm_index]:
                prefix=f'pass{pass_index}/{scope.name}';stage=lambda label,fn:s(prefix+'/'+label,fn)
                bank=build_bank(engine,scope,stage)
                transaction=reserve(engine,native_module,bank,arm,stage)
                before=stage('Native-before-complete-local-trace',lambda:native(engine));protocol=None;records=[]
                try:
                    for cycle,end in enumerate(scope.chunk_ends):
                        one=lambda label,fn:stage(f'cycle{cycle}/'+label,fn)
                        cycle_started=time.perf_counter() # Original diagnostic cycle-clock interval, no cost use.
                        work=generate(engine,native_module,bank,ordinal,one);ordinal+=1
                        def proposals():
                            nonlocal protocol
                            if protocol is None:protocol=setup_normal_cycle(native_module,work['source'],work['plans'],
                                work['rows'],work['confidences'],compiled_protocol=compiled)
                            attach_normal_cycle_proposals(protocol,work['source'],work['plans'],work['rows'],
                                work['confidences'],cycle_id=cycle)
                            work['work_plan'].require_published_tickets(protocol['tickets'])
                        one('real-typed-publication-before-target',proposals)
                        chunk=begin(engine,bank,work,arm,transaction,cycle,one)
                        dispatch_before=one('target-MC2-before',snapshot_mc2_dispatch_counters)
                        candidate=qualify(engine,bank,work,arm,chunk.start,one)
                        oracle=reference(engine,bank,work,arm,chunk.start,candidate,one)
                        if cycle==0 and scope.initial_prompt_cursor==0:
                            snapshot=one('real-normal-cross-M-snapshot',lambda:normal_cross_M_snapshot(engine,candidate))
                            if arm_index==0:cross[pass_index]=snapshot
                            else:
                                comparison=one('real-normal-cross-M-observation',lambda:normal_cross_M_compare(engine,cross.pop(pass_index),snapshot))
                                report.setdefault('cross_M_real_normal',[]).append(dict(pass_index=pass_index,comparison=comparison))
                                one('retain-cross-M-real-normal',lambda:save(directory,f'rank{engine.rank}-pass{pass_index}-cross-M.json.gz',comparison))
                        dispatch_after=one('target-MC2-after',snapshot_mc2_dispatch_counters)
                        def dispatch_gate():
                            values={k:dispatch_after[k]-dispatch_before[k] for k in dispatch_before}
                            require(len(values)==11 and all(v>=0 for v in values.values())
                                and values['fallback']==values['exception']==0,'original target MC2 failure/exception gate')
                            if not engine.is_draft and arm.M==197:
                                require(values['native_epilogue_success']>0,'original197 positive actual native MC2 route')
                            if not engine.is_draft and arm.M==645:
                                require(not report['profile_scope']['profile_M645_membership_only'],
                                    'this epoch specifically uses original unprofiled645 subset, not an unreviewed new profile')
                            return dict(before=dispatch_before,after=dispatch_after,delta=values,
                                MC2_645_qualified=False,actual_M645_route='original-unfused-production' if arm.M==645 else None)
                        dispatch=one('original-MC2-gate-with-unfused645-subset',dispatch_gate)
                        prompt_proof=prepare(engine,bank,transaction,chunk,candidate,one)
                        record=run_original_tree_cycle(engine,native_module,work['source'],work['plans'],work['rows'],
                            None if engine.is_draft else candidate['output']['tree'],protocol,cycle_id=cycle,
                            target_home=cycle%2,stage_all4=one)
                        own=one('own-reference-normal-commit',lambda:commit_reference(engine,bank,work,oracle,record))
                        prompt_commit=commit(engine,bank,transaction,chunk,protocol,record,cycle%2,one)
                        clock=original_clock(engine,transaction,protocol,cycle_started,one)
                        publication=publish(engine,native_module,bank,transaction,one) if chunk.final else None
                        row=dict(cycle=cycle,source=work['source'],selected=work['selected'],rows=work['rows'],
                            confidences=work['confidences'],draft=work['draft_proof'],geometry=arm.geometry(chunk.start),
                            target=None if engine.is_draft else candidate['report'],
                            reference=None if engine.is_draft else oracle['report'],MC2=dispatch,
                            prompt=prompt_proof,prompt_commit=prompt_commit,compiled_normal_commit=record,
                            own_reference_commit=own,clock=clock,publication=publication)
                        records.append(row)
                        one('retain-completed-real-phase',lambda:save(directory,
                            f'rank{engine.rank}-pass{pass_index}-{scope.name}-cycle{cycle}.json.gz',row))
                        del candidate,oracle,work
                    next_work=generate(engine,native_module,bank,ordinal,lambda label,fn:stage('next-draft/'+label,fn));ordinal+=1
                    next_record=dict(source=next_work['source'],rows=next_work['rows'],selected=next_work['selected'],
                        draft=next_work['draft_proof'])
                    abort=None if scope.initial_prompt_cursor else abort_partial(engine,bank,transaction,stage)
                    stable=dict(phases=[dict(source=r['source'],rows=r['rows'],selected=r['selected'],
                        emitted=r['compiled_normal_commit']['committed_tokens'],accepted=r['compiled_normal_commit']['accepted_paths'],
                        final_token=None if r['publication'] is None else r['publication']['first_token']) for r in records],
                        next_draft={k:next_record[k] for k in ('source','rows','selected')})
                    def stable_gate():
                        if zero:require(stable==expected[scope.name],'complete fresh-history zero reproduces all true grants/tokens/paths/publications/nextdraft')
                        else:expected[scope.name]=stable
                    stage('complete-zero-stability',stable_gate)
                    trace_native=stage('complete-trace-Native-delta',lambda:delta(before,native(engine),zero=zero))
                    value=dict(pass_index=pass_index,scope=scope.name,arm=arm.name,zero=zero,
                        committed_prompt_phases=len(records),Native=trace_native,
                        lease=bank.prefill,actual_page_plan=176,records=records,next_draft=next_record,
                        controlled_partial_abort=abort,fingerprint=digest(stable),source_initial_cursor=scope.initial_prompt_cursor,
                        original_prompt_boundaries=list(transaction['boundary'].events))
                    report['traces'].append(value)
                    stage('retain-complete-local-trace',lambda:save(directory,
                        f'rank{engine.rank}-pass{pass_index}-{scope.name}-trace.json.gz',value))
                finally:
                    # A role-local manager state must never select a collective lane.
                    rows=stage4(engine,events,prefix+'/cleanup-reservation-state',
                        lambda:transaction['manager'].current is not None,share=True)
                    active=[r['evidence'] for r in rows]
                    stage('cleanup-reservation-uniform',lambda:require(len(set(active))==1,
                        'all four roles choose the same original abort lane'))
                    if active[0]:abort_partial(engine,bank,transaction,stage)
                    stage('original-boundaries-close',transaction['boundary'].close)
                    stage('release-real-local-bank',engine._release_cache)
    report['inventory_before_seal']=s('complete-own-inventory-before-seal',lambda:inventory(engine))
    s('own-original-Native-seal',engine.seal_graph_cache)
    report['sealed']=bool(engine.graph_runner.graph_cache_sealed)
    s('actual-original-seal-gate',lambda:require(report['sealed'],'actual original own Native graph cache seal'))
    report.update(completed=True,passed_subset=True,Native_numerical_subset_qualified=True,
        MC2_645_qualified=False,adopted=False,complete_local_zero_sweeps=2,
        exploration_prompt_phases=6,fixed_zero_prompt_phases=12,full_component_MC2_qualified=False,
        component_only_not_full527_or_production=True,final_Native=s('final-Native',lambda:native(engine)))
    report['final_inventory']=s('final-own-inventory',lambda:inventory(engine))
    report['boundary_events']=events;report['cache_released']=engine.cache_allocation is None
    return report
