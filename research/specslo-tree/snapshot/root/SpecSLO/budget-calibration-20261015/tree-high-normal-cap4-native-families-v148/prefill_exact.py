"""Target prefill graphs for selected exact shapes; no real or scratch padding."""
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if ROOT.name == 'staged-prefill':
    ROOT = ROOT.parent


def install(native):
    import torch
    examples=json.loads((ROOT/'prompt-shape-examples.json').read_text())['examples']
    allowed=frozenset(int(row['query_count']) for row in examples)
    if len(allowed)!=201 or any(not 1<=n<=1024 for n in allowed):
        raise RuntimeError('Expanded prefill inventory is outside its qualified scope')
    cls = native.NativePearlEngine
    prefill = cls._prefill_and_sample_target_batch
    packed = cls._run_packed_hidden
    metrics = cls.graph_metrics
    seal = cls.seal_graph_cache
    runner_cls = native.NativeACLGraphRunner
    record = runner_cls._record_execution

    def tracked(runner,kind,execution,output,*,entry_key=None,token_rows=()):
        expected=getattr(runner,'_dynamic_exact_prefill_expected_rows',None)
        if expected is not None:
            if (kind!='target' or tuple(token_rows)!=(expected,) or entry_key is None
                    or not entry_key[0].startswith('target-hidden:0|steps:1|fia:')):
                raise RuntimeError('Exact prefill accounting received a non-causal/non-prefill graph')
            if execution.mode=='replay':
                histogram=getattr(runner,'_dynamic_exact_prefill_replays',None)
                if histogram is None:
                    histogram=runner._dynamic_exact_prefill_replays=Counter()
                histogram[expected]+=1
        return record(runner,kind,execution,output,entry_key=entry_key,token_rows=token_rows)
    runner_cls._record_execution=tracked

    def graph(engine,tensor,pos,metadata):
        runner=engine.graph_runner
        previous=getattr(runner,'_dynamic_exact_prefill_expected_rows',None)
        runner._dynamic_exact_prefill_expected_rows=int(tensor.shape[0])
        try:
            return runner.run_tree_hidden(tensor,pos,metadata)
        finally:
            runner._dynamic_exact_prefill_expected_rows=previous

    def active(engine, *args, **kwargs):
        previous = getattr(engine, '_dynamic_prefill_active', False)
        engine._dynamic_prefill_active = True
        try:
            return prefill(engine,*args,**kwargs)
        finally:
            engine._dynamic_prefill_active = previous
    cls._prefill_and_sample_target_batch=active

    def execute(engine, tokens, ids, positions, use_aclgraph=True,
                logit_indices=None, use_fused_infer_attention=False):
        shape = (len(tokens),len(set(ids)))
        admitted = (not engine.is_draft and getattr(engine,'_dynamic_prefill_active',False)
                    and use_fused_infer_attention and shape[1]==1 and len(tokens) in allowed)
        if not admitted:
            return packed(engine,tokens,ids,positions,use_aclgraph,logit_indices,use_fused_infer_attention)
        counts = getattr(engine,'_dynamic_exact_prefill_counts',None)
        if counts is None:
            counts=engine._dynamic_exact_prefill_counts=Counter()
        qualified=getattr(engine,'_dynamic_exact_prefill_qualified',None)
        if qualified is None:
            qualified=engine._dynamic_exact_prefill_qualified=set()
        tensor=torch.tensor(tokens,dtype=torch.long,device=engine.device)
        pos,metadata=engine._prepare_attention_metadata(ids,positions,True)
        hidden=graph(engine,tensor,pos,metadata)
        if shape not in qualified:
            if engine.graph_runner.graph_cache_sealed:
                raise RuntimeError('Unqualified exact prefill extent after sealing')
            torch.npu.synchronize()
            actual=hidden.clone()
            reference=packed(engine,tokens,ids,positions,False,None,True)
            torch.npu.synchronize()
            if not engine.graph_runner._target_outputs_match((actual,),(reference,)):
                raise RuntimeError('Exact prefill graph differs from original eager hidden states')
            if not torch.equal(engine.model.compute_greedy_tokens(actual[[-1]],engine.draft_vocab_size),
                               engine.model.compute_greedy_tokens(reference[[-1]],engine.draft_vocab_size)):
                raise RuntimeError('Exact prefill graph changed the original eager frontier token')
            # Warmup only: force the ORIGINAL changed-input numerical gate to
            # run even for a shape visited by one identical prompt per trace.
            alternate=tensor.clone()
            alternate[-1]=(int(tokens[-1])+1)%engine.draft_vocab_size
            graph(engine,alternate,pos,metadata)
            torch.npu.synchronize()
            hidden=graph(engine,tensor,pos,metadata)
            torch.npu.synchronize()
            if engine.graph_runner.last_target_execution.mode not in ('replay','capture_replay'):
                raise RuntimeError('Exact prefill failed native graph qualification')
            qualified.add(shape)
            counts['original_eager_numerical_qualifications']+=1
        counts['graph_calls']+=1
        counts['real_queries']+=len(tokens)
        counts['physical_queries']+=len(tokens)
        return hidden if logit_indices is None else hidden[logit_indices]
    cls._run_packed_hidden=execute

    def measured(engine):
        histogram=getattr(engine.graph_runner,'_dynamic_exact_prefill_replays',{})
        before=getattr(engine,'_dynamic_exact_prefill_replay_baseline',{})
        return metrics(engine)|{'dynamic_tree_exact_prefill_enabled':int(not engine.is_draft),
            **{'dynamic_tree_exact_prefill_'+key:int(value) for key,value in
               getattr(engine,'_dynamic_exact_prefill_counts',{}).items()},
            **{f'dynamic_tree_measured_exact_prefill_rows_M{rows}':int(value-before.get(rows,0))
               for rows,value in histogram.items()}}
    cls.graph_metrics=measured

    def counted_seal(engine):
        result=seal(engine)
        engine._dynamic_exact_prefill_replay_baseline=dict(
            getattr(engine.graph_runner,'_dynamic_exact_prefill_replays',{}))
        return result
    cls.seal_graph_cache=counted_seal
