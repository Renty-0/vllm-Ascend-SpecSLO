"""Fresh same-instance fixed FULL M120 model and own-history protocol."""
import ast
import copy

import reference218 as old


def target_step(engine,data):
    import hashlib,torch
    from metadata346 import pack_q5
    from scope346 import current
    current(engine)
    old.need(engine._branch_tail_state.enabled is False,'independent reference never uses candidate branch copies')
    engine._prepare_attention_metadata(data['ids'],data['positions'],True)
    inputs,positions,meta=pack_q5(engine,dict(data,selected=data['ids'][:120:5]),full=True)
    signature=((tuple(meta.tree_attention_mask.shape),tuple(meta.request_block_tables.shape)),)
    key=(f'target-greedy:{engine.draft_vocab_size}|steps:1|tree-fia-full:exact-length-task-update|buffers:{signature}',(120,))
    runner=engine.graph_runner;entry=runner.target_entries.get(key)
    old.need(runner.graph_cache_sealed is True and entry is not None and entry.runtime_validated is True
             and key not in runner.disabled_entry_keys,'naturally validated sealed static q5 reference graph')
    old.need(meta.tree_attention and meta.use_fused_infer_attention
             and tuple(meta.actual_seq_lengths_q)==tuple(data['actual_seq_lengths_q'])
             and tuple(meta.sequence_lens)==(2560,)*24
             and tuple(meta.request_block_tables.shape)==(24,32)
             and tuple(meta.tree_attention_mask.shape)==(24,1,10,4096),'actual static q5 reference metadata')
    output=runner.run_target_greedy([inputs],[positions],[meta],engine.draft_vocab_size)[0]
    values=output.cpu().tolist()
    old.need(len(values)==120 and all(type(v)is int and 0<=v<engine.draft_vocab_size for v in values)
             and runner.target_entries[key]is entry,'complete original greedy head and same sealed entry')
    mask_bytes=meta.tree_attention_mask.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return dict(heads80=values,real_tokens=[values[i]for i in data['real_head_rows']],
                metadata=dict(positions=positions.cpu().tolist(),slots=meta.slot_mapping.cpu().tolist(),
                              contexts=meta.context_lens.cpu().tolist(),block_tables=meta.request_block_tables.cpu().tolist(),
                              actual_seq_lengths_q=list(meta.actual_seq_lengths_q),sequence_lens=list(meta.sequence_lens),
                              backend='static_FULL_q5_M120_S24_KV2560',tree_mask_shape=list(meta.tree_attention_mask.shape),
                              zero_page=engine._branch_tail_state.static_zero_page,
                              exact_mask_sha256=hashlib.sha256(mask_bytes).hexdigest(),
                              future_query_visibility='nonreal q1..q4 attend prefix key0 only'))


def run(*args, **kwargs):
    source = old.Path(old.__file__).read_text()
    before = "backend='original target TP3 BF16 causal FIA Q80/S16; only first query per owner consumed'"
    after = "backend='original target TP3 BF16 static FULL q5 M120/S24 KV2560; only first query per owner consumed'"
    if source.count(before) != 1:
        raise RuntimeError('ROOT346 exact independent reference backend descriptor')
    source = source.replace(before, after)
    source = source.replace('all(0<len(group)<=32 for group in groups)', 'all(0<len(group)<=48 for group in groups)')
    nodes = [copy.deepcopy(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'run']
    namespace = dict(vars(old), target_step=target_step, frame=frame)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 __file__+'::original-independent-history-loop', 'exec'), namespace)
    return namespace['run'](*args, **kwargs)


def frame(prompts,outputs,indices,*,future=0):
    data=old.frame(prompts,outputs,indices,future=future)
    # Keep the original16 real-owner histories and append eight independent
    # nonreal q5 groups, without a short tail.
    used=16-len(indices)
    for j in range(used,used+8):
        data['tokens'].extend([0]+[future]*4);data['ids'].extend([463+j]*5)
        data['positions'].extend(range(5));data['sequence_lens'].append(5)
    data['actual_seq_lengths_q']=list(range(5,121,5))
    if len(data['tokens'])!=120 or len(data['sequence_lens'])!=24:raise RuntimeError('original real own-history frame with explicit120 physical rows')
    return data
