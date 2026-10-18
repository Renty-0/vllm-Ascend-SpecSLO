"""Process-local tree budget policy; use existing tree/KV correctness protocol."""
import inspect
import hashlib
from functools import partial
import json
import os
from pathlib import Path
import sys
import textwrap
import time

P = Path(__file__).resolve().parent


def patched_source(original):
    source = textwrap.dedent(original)
    before = '    request_params = list(request_params)'
    assert source.count(before)==1
    source = source.replace(before,
        '    request_params = [replace(p, spec_rhythm_max_gamma=5) for p in request_params]')
    before = 'max_candidates=min(cap, tree_plan.width * tree_plan.depth),'
    assert source.count(before)==1
    return source.replace(before,'max_candidates=min(cap, tree_plan.candidate_budget),')


def npu_memory(engine):
    if getattr(getattr(engine,'device',None),'type',None)!='npu':
        return {}
    import torch
    return {'npu_allocated_bytes':int(torch.npu.memory_allocated(engine.device)),
            'npu_reserved_bytes':int(torch.npu.memory_reserved(engine.device)),
            'npu_peak_allocated_bytes':int(torch.npu.max_memory_allocated(engine.device))}


def install(*,fast=False,host=False,transport=False,replay=False,repair=False,prefill=False):
    from launch_readiness import require_startup_scope
    require_startup_scope()
    if prefill:
        from qualification_prerequisites import require
        require()
    from vllm_ascend.spec_decode.pearl import native_engine as native
    from vllm_ascend.spec_decode.pearl import spec_rhythm as rhythm
    from budget_policy import TreeBudgetPolicy
    policy = TreeBudgetPolicy(json.loads((P/'tree-gain-model.json').read_text()))
    if replay:
        import tree_target_replay
        from vllm_ascend.spec_decode.pearl import native_graph
        tree_target_replay.install(native_graph)
        original_init=native.NativePearlEngine.__init__
        def initialized(engine,*args,**kwargs):
            original_init(engine,*args,**kwargs)
            engine.graph_runner._dynamic_tree_target_replay_first=not engine.is_draft
        native.NativePearlEngine.__init__=initialized
    if repair:
        import root_seed
        import parallel_steps
        from vllm_ascend.spec_decode.pearl import native_graph
        root_seed.install(native)
        parallel_steps.install(native_graph)
    elif fast:
        import fast_tree_draft
        fast_tree_draft.install(native)
    if host:
        import host_target_metadata
        host_target_metadata.install(native)
    if transport:
        import host_tree_transport
        host_tree_transport.install(native)
    import verification_qualification
    verification_qualification.install(native)
    import frontier_qualification
    frontier_qualification.install(native)
    if prefill:
        import prefill_exact
        from prefill_inventory import qualify as qualify_prefill
        prefill_exact.install(native)
        from vllm_ascend.spec_decode.pearl import native_graph,native_model
        import prefill_arena
        prefill_arena.install(native_graph,native_model)
        examples=json.loads((P/'prompt-shape-examples.json').read_text())['examples']
        before_initialization=native.NativePearlEngine.__init__
        def with_inventory(engine,*args,**kwargs):
            before_initialization(engine,*args,**kwargs)
            from mc2_inventory_contract import preflight as preflight_MC2
            preflight_MC2(engine)
            if engine.is_draft:
                # The unchanged native generic-tree qualifier closes its
                # largest request bucket to 2*max_num_seqs, each with depth+1
                # scratch queries (64*9=576 here). This is an untimed scratch
                # capacity, separate from actual verification's 128-query cap.
                rows=1 << int(engine.config.max_num_seqs).bit_length()
                depth=int(engine.config.spec_rhythm_tree_depth)
                width=int(engine.config.spec_rhythm_tree_width)
                cap=max(depth+1,width*depth-depth+1)
                engine.graph_runner.max_graph_tokens=max(engine.graph_runner.max_graph_tokens,rows*cap)
            if not engine.is_draft:
                largest=max(row['query_count'] for row in examples)
                if largest>min(1024,engine.config.max_model_len):
                    raise RuntimeError('Expanded exact prefill exceeds the configured context scope')
                engine.graph_runner.max_graph_tokens=max(engine.graph_runner.max_graph_tokens,largest)
            # Complete this before the API publishes worker readiness and
            # before the benchmark materializes any request arrival clock.
            engine._allocate_cache([],enable_prefix_caching=False,reserve_sequence_capacity=True)
            try:
                import torch
                with torch.inference_mode():
                    for sequence in engine._tree_graph_scratch_sequence_ids:
                        engine._activate_cache_sequence(sequence,[0],enable_prefix_caching=False)
                    qualify_prefill(engine,examples)
                    from target_inventory import qualify as qualify_target
                    qualify_target(engine)
                    from draft_inventory import qualify as qualify_draft
                    qualify_draft(engine)
            finally:
                engine._release_cache()
        native.NativePearlEngine.__init__=with_inventory
    shaper = rhythm.SpecRhythmBudgetShaper
    original_shape = shaper.shape

    def shape(instance, **kwargs):
        normal,eager = policy.allocate(kwargs['normal_request_indices'],
            kwargs['eager_request_indices'],kwargs['states'],kwargs['projected_wait_ms'],
            kwargs.get('draft_token_budget'))
        gaps = {i:kwargs['states'][i].projected_progress_gap(kwargs['projected_wait_ms'])
                for i in (*kwargs['normal_request_indices'],*kwargs['eager_request_indices'])}
        return rhythm.SpecRhythmBudgetPlan(
            plan_id=kwargs['plan_id'],normal_budgets=normal,eager_budgets=eager,
            progress_gaps=gaps,eager_priorities={i:float(gaps[i]) for i in eager},
            verification_roof=policy.candidate_roof,
            draft_token_budget=kwargs.get('draft_token_budget') or 2*policy.candidate_roof,
            allocated_draft_tokens=sum(normal.values())+sum(eager.values()),
            deferred_normal_request_indices=tuple(i for i in kwargs['normal_request_indices'] if i not in normal))
    shaper.shape = shape
    cls = native.NativePearlEngine
    source = patched_source(inspect.getsource(cls._generate_spec_rhythm_tree_decode))
    import canonical_eager
    canonical_eager.install(native)
    source = canonical_eager.patch_decode_source(source)
    import backbone_selection
    vars(native)["_dt_select_backbone"]=backbone_selection.select
    source=backbone_selection.patch_decode(source)
    from packed_window import PackedTreeWindowEstimator
    vars(native)['_dt_PackedTreeWindowEstimator']=PackedTreeWindowEstimator
    before='window_estimator = DraftWindowEstimator(self.config.spec_rhythm_acceptance_ema_alpha)'
    assert source.count(before)==1
    source=source.replace(before,
        'window_estimator = _dt_PackedTreeWindowEstimator(self.config.spec_rhythm_acceptance_ema_alpha)\n'
        '    self._dynamic_tree_window_estimator = window_estimator')
    before='eager_work=bool(eager_budgets))'
    assert source.count(before)==1
    source=source.replace(before,'eager_work=bool(eager_budgets), eager_rows=len(eager_budgets))')
    if transport:
        source=host_tree_transport.patch_decode_source(source)
    import early_target
    vars(native)['_dt_tree_early_target_eligible']=early_target.eligible
    vars(native)['_dt_tree_early_target_decide']=early_target.decide
    vars(native)['_dt_tree_early_target_record']=early_target.record
    source=early_target.patch_decode(source)
    runtime_pin_path=P/'RUNTIME_SOURCE_PIN.json'
    if runtime_pin_path.exists():
        expected=json.loads(runtime_pin_path.read_text())['sha256']
        if hashlib.sha256(source.encode()).hexdigest()!=expected:
            raise RuntimeError('Generated V36 runtime source differs from its reviewed pin')
    elif prefill:
        raise RuntimeError('V36 formal runtime source must be prepared and pinned before model startup')
    (P/'runtime_tree_decode.py').write_text(source)
    namespace = vars(native)
    exec(compile(source,str(P/'runtime_tree_decode.py'),'exec'),namespace)
    decode_impl = namespace['_generate_spec_rhythm_tree_decode']
    def fresh_decode(engine,**kwargs):
        if int(policy.model.get('tree_depth',0))!=int(engine.config.spec_rhythm_tree_depth):
            raise RuntimeError('Tree depth and calibrated gain model disagree')
        # Warmup may discover graph shapes, but its future workload outcomes
        # are not carried into the measured policy's online gain estimates.
        policy.observed.clear()
        engine._dynamic_tree_current_target_max_queries=0
        generation=getattr(engine,'_dynamic_tree_generation',0)+1
        engine._dynamic_tree_generation=generation
        runner=getattr(engine,'graph_runner',None)
        device=getattr(engine,'device',None)
        track=getattr(device,'type',None)=='npu'
        started=time.perf_counter()
        if track:
            progress_dir=P/'generation-progress'
            progress_dir.mkdir(exist_ok=True)
            progress_path=progress_dir/f'rank{engine.rank}-pid{os.getpid()}.json'
            progress={'rank':engine.rank,'pid':os.getpid(),'generation':generation,
                      'sealed':bool(getattr(runner,'graph_cache_sealed',False)),
                      'status':'running','started_monotonic':started,
                      'memory_before':npu_memory(engine)}
            progress_path.write_text(json.dumps(progress,indent=2)+'\n')
        try:
            result=decode_impl(engine,**kwargs)
            if track:
                progress.update(status='complete',elapsed_seconds=time.perf_counter()-started,
                    current_target_max_queries=engine._dynamic_tree_current_target_max_queries,
                    phase_seconds=getattr(engine,'last_worker_decode_phase_seconds',{}),
                    decode_counters=dict(getattr(engine,'last_worker_decode_counters',{})),
                    CPU_control_counts=dict(getattr(engine,'_dynamic_tree_cpu_counts',{})),
                    graph_metrics={k:v for k,v in engine.graph_metrics().items() if k.startswith('aclgraph_')},
                    graph_inventory={kind:[repr(key) for key in getattr(runner,name,{}).keys()] for kind,name in [('generic','entries'),('draft','draft_entries'),('target','target_entries')]})
            return result
        except BaseException:
            if track:
                progress.update(status='failed',elapsed_seconds=time.perf_counter()-started)
            raise
        finally:
            if track:
                progress['memory_after']=npu_memory(engine)
                progress_path.write_text(json.dumps(progress,indent=2)+'\n')
                history=progress_dir/f'rank{engine.rank}-pid{os.getpid()}-history.jsonl'
                with history.open('a') as output:
                    output.write(json.dumps(progress)+'\n')
    cls._generate_spec_rhythm_tree_decode = fresh_decode
    from target_layout import pad_target
    cls._pad_spec_rhythm_target_tree_graph=pad_target
    record = rhythm.SpecRhythmRuntimeState.record_verification
    def observed(state, **kwargs):
        record(state,**kwargs)
        policy.observe(state,int(kwargs['proposed_tokens']),int(kwargs['delivered_tokens']))
    rhythm.SpecRhythmRuntimeState.record_verification = observed
    target = cls.target_tree_forward
    def account_real_target_plans(engine, plans):
        """Account only real tree work, also used by the mixed target bridge.

        Mixed prompt and private padding rows do not consume the speculative
        verification budget. Every original role executes this same host
        accounting once; verification outcomes still update at record_verification.
        """
        queries=policy.check_target([p.candidate_budget for p in plans])
        engine._dynamic_tree_current_target_max_queries=max(
            getattr(engine,'_dynamic_tree_current_target_max_queries',0),queries)
        for plan in plans:
            parents=plan.parent_indices.detach().cpu().tolist()
            fork=any(parents.count(parent)>1 for parent in set(parents))
            policy.counts['target_branching_proposals' if fork else 'target_linear_proposals']+=1
            policy.counts[f'actual_target_B{plan.candidate_budget}']+=1
    cls._specslo_account_real_target_plans = account_real_target_plans
    def checked_target(engine,plans,*args,**kwargs):
        real_count = kwargs.get('real_tree_count') or len(plans)
        account_real_target_plans(engine, plans[:real_count])
        return target(engine,plans,*args,**kwargs)
    cls.target_tree_forward = checked_target
    metrics = cls.graph_metrics
    seal=cls.seal_graph_cache
    baseline={}
    fast_baseline={}
    host_baseline=[0]
    def measured_seal(engine):
        result=seal(engine)
        baseline.clear();baseline.update(policy.counts)
        fast_baseline.clear();fast_baseline.update(getattr(engine,'_dynamic_tree_draft_counts',{}))
        host_baseline[0]=getattr(engine,'_dynamic_tree_host_target_calls',0)
        return result
    cls.seal_graph_cache=measured_seal
    def measured_metrics(engine):
        window=getattr(engine,'_dynamic_tree_window_estimator',None)
        return metrics(engine)|npu_memory(engine)|{
            'finite_target_inventory_completed':int(getattr(engine,'_finite_target_inventory_complete',False)),
            'finite_draft_inventory_completed':int(getattr(engine,'_finite_draft_inventory_complete',False)),
            **{'dynamic_tree_window_'+k:v for k,v in (window.summary() if window else {}).items()},
            'dynamic_tree_canonical_eager_enabled':int(getattr(engine,'_dynamic_tree_canonical_eager',False)),
            'dynamic_tree_global_target_queries':policy.target_queries,
            'dynamic_tree_candidate_roof':policy.candidate_roof,
            **{'dynamic_tree_'+k:int(v) for k,v in policy.counts.items()},
            **{'dynamic_tree_measured_'+k:int(v-baseline.get(k,0)) for k,v in policy.counts.items()},
            **{'dynamic_tree_fast_draft_'+k:int(v) for k,v in
                getattr(engine,'_dynamic_tree_draft_counts',{}).items()},
            **{'dynamic_tree_measured_fast_draft_'+k:int(v-fast_baseline.get(k,0)) for k,v in
                getattr(engine,'_dynamic_tree_draft_counts',{}).items()},
            'dynamic_tree_measured_target_max_actual_queries':getattr(engine,'_dynamic_tree_current_target_max_queries',0),
            'dynamic_tree_host_target_metadata_calls':getattr(engine,'_dynamic_tree_host_target_calls',0),
            'dynamic_tree_measured_host_target_metadata_calls':getattr(engine,'_dynamic_tree_host_target_calls',0)-host_baseline[0],
            **{f'dynamic_tree_current_generation_gain_{c}_B{n}_rounds':int(v[0])
               for (c,n),v in policy.observed.items()},
            **{f'dynamic_tree_current_generation_gain_{c}_B{n}_delivered':int(v[1])
               for (c,n),v in policy.observed.items()}}
    cls.graph_metrics = measured_metrics
    return policy


def tree_worker(*args,fast=False,host=False,transport=False,replay=False,repair=False,prefill=False,**kwargs):
    from vllm_ascend.spec_decode.pearl import api
    original = api._pearl_worker
    install(fast=fast,host=host,transport=transport,replay=replay,repair=repair,prefill=prefill)
    return original(*args,**kwargs)


def main():
    from launch_readiness import require_launch
    require_launch()
    from qualification_prerequisites import require
    require()
    from vllm_ascend.spec_decode.pearl import api
    from examples.benchmark_nano_pearl_speculative import main as benchmark
    fast='--tree-fast-draft' in sys.argv
    if fast:
        sys.argv.remove('--tree-fast-draft')
    host='--tree-host-metadata' in sys.argv
    if host:
        sys.argv.remove('--tree-host-metadata')
    transport='--tree-host-transport' in sys.argv
    if transport:
        sys.argv.remove('--tree-host-transport')
    replay='--tree-target-replay-first' in sys.argv
    if replay:
        sys.argv.remove('--tree-target-replay-first')
    repair='--tree-draft-pa-repair' in sys.argv
    if repair:
        sys.argv.remove('--tree-draft-pa-repair')
    prefill='--tree-exact-prefill' in sys.argv
    if prefill:
        sys.argv.remove('--tree-exact-prefill')
    api._pearl_worker = partial(tree_worker,fast=fast,host=host,transport=transport,replay=replay,repair=repair,prefill=prefill)
    benchmark(sys.argv[1:])


if __name__=='__main__':
    from launch_guard import require_online_qualification as _v75_block_old_cli
    _v75_block_old_cli()
    main()
