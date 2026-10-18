"""Same independent218 own-output histories, using canonical FULL9 M80."""
import ast
import copy

import reference218 as old


def target_step(engine,data):
    import hashlib,torch
    from static_reference246 import pack_q5
    from scope245 import current
    current(engine)
    old.need(engine._branch_tail_state.enabled is False,'independent reference never uses candidate branch copies')
    engine._prepare_attention_metadata(data['ids'],data['positions'],True)
    inputs,positions,meta=pack_q5(engine,dict(data,selected=data['ids'][::5]),full=True)
    signature=((tuple(meta.tree_attention_mask.shape),tuple(meta.request_block_tables.shape)),)
    key=(f'target-greedy:{engine.draft_vocab_size}|steps:1|tree-fia-full:exact-length-task-update|buffers:{signature}',(80,))
    runner=engine.graph_runner;entry=runner.target_entries.get(key)
    old.need(runner.graph_cache_sealed is True and entry is not None and entry.runtime_validated is True
             and key not in runner.disabled_entry_keys,'naturally validated sealed static q5 reference graph')
    old.need(meta.tree_attention and meta.use_fused_infer_attention
             and tuple(meta.actual_seq_lengths_q)==tuple(data['actual_seq_lengths_q'])
             and tuple(meta.sequence_lens)==(2560,)*16
             and tuple(meta.request_block_tables.shape)==(16,32)
             and tuple(meta.tree_attention_mask.shape)==(16,1,10,4096),'actual static q5 reference metadata')
    output=runner.run_target_greedy([inputs],[positions],[meta],engine.draft_vocab_size)[0]
    values=output.cpu().tolist()
    old.need(len(values)==80 and all(type(v)is int and 0<=v<engine.draft_vocab_size for v in values)
             and runner.target_entries[key]is entry,'complete original greedy head and same sealed entry')
    mask_bytes=meta.tree_attention_mask.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return dict(heads80=values,real_tokens=[values[i]for i in data['real_head_rows']],
                metadata=dict(positions=positions.cpu().tolist(),slots=meta.slot_mapping.cpu().tolist(),
                              contexts=meta.context_lens.cpu().tolist(),block_tables=meta.request_block_tables.cpu().tolist(),
                              actual_seq_lengths_q=list(meta.actual_seq_lengths_q),sequence_lens=list(meta.sequence_lens),
                              backend='static_FULL_q5_M80_S16_KV2560',tree_mask_shape=list(meta.tree_attention_mask.shape),
                              zero_page=engine._branch_tail_state.static_zero_page,
                              exact_mask_sha256=hashlib.sha256(mask_bytes).hexdigest(),
                              future_query_visibility='nonreal q1..q4 attend prefix key0 only'))


def run(*args, **kwargs):
    source = old.Path(old.__file__).read_text()
    before = "backend='original target TP3 BF16 causal FIA Q80/S16; only first query per owner consumed'"
    after = "backend='original target TP3 BF16 static FULL q5 M80/S16 KV2560; only first query per owner consumed'"
    if source.count(before) != 1:
        raise RuntimeError('ROOT245 exact independent reference backend descriptor')
    source = source.replace(before, after)
    nodes = [copy.deepcopy(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'run']
    namespace = dict(vars(old), target_step=target_step)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 __file__+'::original-independent-history-loop', 'exec'), namespace)
    return namespace['run'](*args, **kwargs)
