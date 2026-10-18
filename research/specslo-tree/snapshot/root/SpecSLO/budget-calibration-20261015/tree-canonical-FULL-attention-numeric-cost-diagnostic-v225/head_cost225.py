"""Isolated validated head ACLGraph ABBA device-event diagnostics, not GP."""
def measure(engine,hidden,stage):
    import torch,statistics
    from precision_ties225 import greedy
    target=not engine.is_draft;cache={}
    for name in('original','refined'):
        def prepare():
            if not target:return None
            fn=(lambda:engine.model.compute_greedy_tokens(hidden,engine.draft_vocab_size))if name=='original'else(lambda:greedy(engine.model.lm_head,hidden,engine.draft_vocab_size)[0])
            for _ in range(3):expected=fn().clone()
            torch.npu.synchronize()
            graph=torch.npu.NPUGraph()
            with torch.npu.graph(graph):output=fn()
            graph.replay();torch.npu.synchronize()
            if not torch.equal(output,expected):raise RuntimeError('head-only natural eager/graph validation '+name)
            for _ in range(2):
                graph.replay();torch.npu.synchronize()
                if not torch.equal(output,expected):raise RuntimeError('head graph repeat '+name)
            cache[name]=(graph,output,expected)
        stage('head-cost/'+name+'/natural-capture-exact-repeat',prepare)
    samples=[]
    for block in range(3):
        for cell,name in enumerate(('original','refined','refined','original')):
            def sample():
                if not target:return None
                start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(20):cache[name][0].replay()
                end.record();torch.npu.synchronize()
                elapsed=float(start.elapsed_time(end))/20
                if not elapsed>0 or not torch.equal(cache[name][1],cache[name][2]):raise RuntimeError('timed head graph stability')
                return dict(block=block,cell=cell,mode=name,replays=20,device_ms_per_replay=elapsed)
            samples.append(stage('head-cost/block'+str(block)+'/cell'+str(cell)+'/'+name,sample))
    if not target:return None
    medians={name:statistics.median(r['device_ms_per_replay']for r in samples if r['mode']==name)for name in cache}
    return dict(M=hidden.shape[0],replay_samples=samples,median_device_ms=medians,
        extra_median_ms=medians['refined']-medians['original'],eager_graph_and_repeat_exact=True,
        scope='isolated captured target LM-head with collective, ABBA device-event timing; not model or Goodput',
        production_graph_flags_modified=False,performance_qualification=False)
