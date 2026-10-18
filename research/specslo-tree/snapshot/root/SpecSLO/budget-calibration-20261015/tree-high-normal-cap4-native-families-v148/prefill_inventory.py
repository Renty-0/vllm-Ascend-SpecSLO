"""Untimed exact prefill inventory plus independent cross-shape restoration."""

def qualify(engine, examples):
    import json,time
    from pathlib import Path
    import torch
    import torch.distributed as dist
    import prefill_arena
    if engine.graph_runner.graph_cache_sealed:
        return
    group=engine.groups.verification_coordination_group
    if group is None or dist.get_backend(group)!='gloo':
        raise RuntimeError('Prefill inventory requires full CPU coordination')
    dist.barrier(group=group)
    if not engine.is_draft:
        seq=engine._tree_graph_scratch_sequence_ids[0]
        previous=getattr(engine,'_dynamic_prefill_active',False)
        progress_dir=Path(__file__).resolve().parent/'startup-progress'
        progress_dir.mkdir(exist_ok=True)
        path=progress_dir/f'prefill-rank{engine.rank}.json'
        history=progress_dir/f'prefill-rank{engine.rank}-history.jsonl'
        progress={'status':'running','completed':0,'total':len(examples)}
        def save():
            progress['updated_unix']=time.time()
            temporary=path.with_suffix('.tmp')
            temporary.write_text(json.dumps(progress,indent=2)+'\n')
            temporary.replace(path)
        def memory():
            workspace=[v for k,v in engine.graph_runner._fia_workspace_pool.items() if k[0]=='causal-fia']
            return {'causal_workspace_buffers':len(workspace),
                    'causal_workspace_bytes':sum(v.numel()*v.element_size() for v in workspace),
                    'npu_allocated_bytes':int(torch.npu.memory_allocated()),
                    'npu_peak_allocated_bytes':int(torch.npu.max_memory_allocated())}
        engine._dynamic_prefill_active=True
        try:
            for index,example in enumerate(sorted(examples,key=lambda row:row['query_count'],reverse=True)):
                progress.update(status='running',completed=index,next_M=example['query_count'])
                save()
                tokens=list(example['token_ids'])
                if len(tokens)!=example['query_count']:
                    raise ValueError('Prefill inventory has inconsistent token extent')
                engine._run_packed_hidden(tokens,[seq]*len(tokens),list(range(len(tokens))),False,[len(tokens)-1],True)
                progress.update(completed=index+1,last_M=example['query_count'],**memory())
                if index%25==0 or index+1==len(examples):
                    progress.update(prefill_arena.retained_task_storage(engine.graph_runner))
                    with history.open('a') as stream:stream.write(json.dumps(progress)+'\n')
                save()
            # Revisit already-qualified larger/smaller graphs with unchanged
            # graph inputs after other shapes overwrite the shared arena.
            # The reference always runs the ORIGINAL eager attention path.
            by_size={row['query_count']:row for row in examples}
            transition_sizes=[1022,59,794,506,753,513,1022,59]
            for number,size in enumerate(transition_sizes):
                tokens=list(by_size[size]['token_ids'])
                ids=[seq]*size;positions=list(range(size))
                engine._dynamic_prefill_active=False
                reference=engine._run_packed_hidden(tokens,ids,positions,False,None,True)
                torch.npu.synchronize()
                engine._dynamic_prefill_active=True
                actual=engine._run_packed_hidden(tokens,ids,positions,False,None,True)
                torch.npu.synchronize()
                if not engine.graph_runner._target_outputs_match((actual,),(reference,)):
                    raise RuntimeError('Shared prefill arena changed hidden states across graph shapes')
                if not torch.equal(engine.model.compute_greedy_tokens(actual[[-1]],engine.draft_vocab_size),
                                   engine.model.compute_greedy_tokens(reference[[-1]],engine.draft_vocab_size)):
                    raise RuntimeError('Shared prefill arena changed greedy tokens across graph shapes')
                if bool(engine._spec_rhythm_nonfinite_flag(include_layer_cache=False).cpu().item()):
                    raise RuntimeError('Shared prefill arena set the numerical-fault flag')
                progress.update(status='transition_checks',transition_checks_passed=number+1,transition_M=size,**memory())
                save()
            progress.update(status='complete',**prefill_arena.retained_task_storage(engine.graph_runner),**memory())
            save()
        except BaseException as error:
            progress.update(status='failed',error_type=type(error).__name__)
            save()
            raise
        finally:
            engine._dynamic_prefill_active=previous
    dist.barrier(group=group)
