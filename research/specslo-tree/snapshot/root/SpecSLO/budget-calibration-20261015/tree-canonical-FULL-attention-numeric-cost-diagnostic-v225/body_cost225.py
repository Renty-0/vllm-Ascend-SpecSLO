"""Original dynamic attention replay wall/device ABBA timings, body only."""
def measure(engine,frame,stage):
    import torch,time,statistics
    from metadata225 import pack
    from dataclasses import asdict
    target=not engine.is_draft;cache={};samples=[]
    shifted=dict(frame,tokens=list(frame['tokens']),positions=[p+1 for p in frame['positions']],sequence_lens=[p+1 for p in frame['sequence_lens']])
    for j in range(16):shifted['tokens'][5*j:5*j+5]=frame['tokens'][5*j+1:5*j+5]+[0]
    for name in('causal','FULL9'):
        def prepare():
            if not target:return None
            for f in(frame,shifted):
                for seq,p in zip(f['ids'],f['positions'],strict=True):engine._ensure_cache_capacity([seq],[p])
            pair=[pack(engine,f,full=name=='FULL9')for f in(frame,shifted)]
            for _ in range(2):
                for inputs,pos,meta in pair:
                    h=engine.graph_runner.run_tree_hidden(inputs,pos,meta)
                    if engine.graph_runner.last_target_execution.mode not in('capture_replay','replay'):raise RuntimeError('original actual body graph required')
                    if not bool(torch.isfinite(h).all()):raise RuntimeError('body finite')
            cache[name]=pair
        stage('dynamic-body-cost/'+name+'/prepare-original-graphs',prepare)
    for block in range(3):
        for cell,name in enumerate(('causal','FULL9','FULL9','causal')):
            def sample():
                if not target:return None
                start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                before=dict(updates=int(getattr(engine.graph_runner,'_dynamic_tree_target_replay_first_updates',0)),skips=engine.graph_runner.task_update_skip_replay_count);host=time.perf_counter();start.record()
                for step in range(6):
                    inputs,pos,meta=cache[name][step%2]
                    h=engine.graph_runner.run_tree_hidden(inputs,pos,meta)
                end.record();torch.npu.synchronize();wall=(time.perf_counter()-host)*1000/6;device=float(start.elapsed_time(end))/6
                if not bool(torch.isfinite(h).all())or engine.graph_runner.last_target_execution.mode!='replay':raise RuntimeError('actual repeated finite body graph')
                after=dict(updates=int(getattr(engine.graph_runner,'_dynamic_tree_target_replay_first_updates',0)),skips=engine.graph_runner.task_update_skip_replay_count)
                delta={k:after[k]-before[k]for k in before}
                return dict(block=block,cell=cell,mode=name,replays=6,host_wall_ms_per_call=wall,device_ms_per_call=device,
                    execution=asdict(engine.graph_runner.last_target_execution),counter_delta=delta)
            samples.append(stage('dynamic-body-cost/block'+str(block)+'/cell'+str(cell)+'/'+name,sample))
    if not target:return None
    medians={name:{clock:statistics.median(r[clock]for r in samples if r['mode']==name)for clock in('host_wall_ms_per_call','device_ms_per_call')}for name in cache}
    return dict(M=80,real_owners=16,alternating_two_adjacent_context_frames=True,samples=samples,medians=medians,
        includes_original_metadata_copy_replay_and_attention_updates=True,model_head_excluded=True,
        no_end_to_end_Goodput_prediction=True,Native_validation_flags_assigned=False)
