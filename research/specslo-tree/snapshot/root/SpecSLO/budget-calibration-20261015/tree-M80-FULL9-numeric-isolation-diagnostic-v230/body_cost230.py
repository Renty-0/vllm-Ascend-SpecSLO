"""Absolute timing of the already naturally qualified q1 graph; no new family."""
def measure(engine,frame,stage):
    import torch,time,statistics
    from metadata230 import pack
    from dataclasses import asdict
    target=not engine.is_draft;cache=[];samples=[];captures=None;qualification=[]
    shifted=dict(frame,tokens=list(frame['tokens']),positions=[p+1 for p in frame['positions']],sequence_lens=[p+1 for p in frame['sequence_lens']])
    for j in range(16):shifted['tokens'][5*j:5*j+5]=frame['tokens'][5*j+1:5*j+5]+[0]
    def prepare():
        nonlocal captures
        if not target:return None
        captures=engine.graph_runner.capture_attempt_count
        for f in(frame,shifted):
            for seq,p in zip(f['ids'],f['positions'],strict=True):engine._ensure_cache_capacity([seq],[p])
            inputs,pos,meta=pack(engine,f,full=True)
            graph=engine.graph_runner.run_tree_hidden(inputs,pos,meta).clone()
            if engine.graph_runner.last_target_execution.mode!='replay':raise RuntimeError('existing q5 FULL9 replay required; never capture for this cost probe')
            eager=engine.model(inputs,pos,meta).clone()
            if not engine.graph_runner._target_outputs_match((graph,),(eager,)):raise RuntimeError('original Native matcher for both actual contexts')
            qualification.append(dict(original_Native_matcher_passed=True,hidden_bitwise_equal=torch.equal(graph,eager)))
            cache.append((inputs,pos,meta))
        if engine.graph_runner.capture_attempt_count!=captures:raise RuntimeError('zero new captures')
    stage('existing-q5-cost/prepare-and-eager-match-two-actual-contexts',prepare)
    for block in range(3):
        for cell in range(2):
            def sample():
                if not target:return None
                start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                before=dict(updates=int(getattr(engine.graph_runner,'_dynamic_tree_target_replay_first_updates',0)),skips=engine.graph_runner.task_update_skip_replay_count)
                host=time.perf_counter();start.record()
                for step in range(6):
                    inputs,pos,meta=cache[step%2];h=engine.graph_runner.run_tree_hidden(inputs,pos,meta)
                end.record();torch.npu.synchronize();wall=(time.perf_counter()-host)*1000/6;device=float(start.elapsed_time(end))/6
                if not bool(torch.isfinite(h).all())or engine.graph_runner.last_target_execution.mode!='replay':raise RuntimeError('actual finite existing graph replay')
                after=dict(updates=int(getattr(engine.graph_runner,'_dynamic_tree_target_replay_first_updates',0)),skips=engine.graph_runner.task_update_skip_replay_count)
                return dict(block=block,cell=cell,mode='q5_FULL9',replays=6,host_wall_ms_per_call=wall,device_ms_per_call=device,execution=asdict(engine.graph_runner.last_target_execution),counter_delta={k:after[k]-before[k]for k in before})
            samples.append(stage(f'existing-q5-cost/block{block}/cell{cell}',sample))
    if not target:return None
    if engine.graph_runner.capture_attempt_count!=captures:raise RuntimeError('zero new graph captures during timing')
    return dict(M=80,real_owners=16,alternating_two_adjacent_context_frames=True,samples=samples,zero_new_body_graph_captures=True,
        eager_graph_qualification=qualification,medians={clock:statistics.median(r[clock]for r in samples)for clock in('host_wall_ms_per_call','device_ms_per_call')},
        model_head_excluded=True,prebuilt_metadata=True,no_end_to_end_Goodput_prediction=True,no_comparative_speed_claim=True)
