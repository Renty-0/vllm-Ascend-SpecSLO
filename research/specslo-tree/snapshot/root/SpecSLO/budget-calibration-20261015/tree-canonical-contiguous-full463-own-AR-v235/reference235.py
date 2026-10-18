"""Same independent218 own-output histories, using canonical FULL9 M80."""
import ast
import copy

import reference218 as old


def target_step(engine, data):
    import torch
    from metadata231 import pack_q5
    from scope235 import current
    current(engine)
    old.need(engine._branch_tail_state.enabled is False, 'reference never uses candidate branch copies')
    # The original helper grows only the reference's own cache. Its causal
    # metadata is discarded; all real model evaluation below uses FULL9.
    engine._prepare_attention_metadata(data['ids'], data['positions'], True)
    frame = dict(data, selected=data['ids'][::5])
    inputs, positions, meta = pack_q5(engine, frame, full=True)
    signature = ((tuple(meta.tree_attention_mask.shape), tuple(meta.request_block_tables.shape)),)
    key = (f'target-greedy:{engine.draft_vocab_size}|steps:1|tree-fia-full:exact-length-task-update|buffers:{signature}', (80,))
    runner = engine.graph_runner
    entry = runner.target_entries.get(key)
    old.need(runner.graph_cache_sealed is True and entry is not None and entry.runtime_validated is True
             and key not in runner.disabled_entry_keys, 'naturally validated and sealed FULL9/S16/M80 entry')
    old.need(meta.use_fused_infer_attention is True and meta.tree_attention is True
             and list(meta.actual_seq_lengths_q) == data['actual_seq_lengths_q']
             and list(meta.sequence_lens) == data['sequence_lens']
             and tuple(meta.block_tables.shape) == tuple(meta.request_block_tables.shape) == (16, 32)
             and tuple(meta.tree_attention_mask.shape) == (16, 1, 9, 4096), 'own exact FULL9 metadata ABI')
    output = runner.run_target_greedy([inputs], [positions], [meta], engine.draft_vocab_size)[0]
    values = output.cpu().tolist()
    old.need(len(values) == 80 and all(type(v) is int and 0 <= v < engine.draft_vocab_size for v in values)
             and runner.target_entries[key] is entry, 'complete80 original greedy head and stable graph identity')
    return dict(heads80=values, real_tokens=[values[i] for i in data['real_head_rows']],
                metadata=dict(positions=positions.cpu().tolist(), slots=meta.slot_mapping.cpu().tolist(),
                              contexts=meta.context_lens.cpu().tolist(), block_tables=meta.request_block_tables.cpu().tolist(),
                              actual_seq_lengths_q=list(meta.actual_seq_lengths_q), sequence_lens=list(meta.sequence_lens),
                              backend='canonical_FULL9_M80_S16', tree_mask_shape=list(meta.tree_attention_mask.shape)))


def run(*args, **kwargs):
    source = old.Path(old.__file__).read_text()
    before = "backend='original target TP3 BF16 causal FIA Q80/S16; only first query per owner consumed'"
    after = "backend='original target TP3 BF16 FULL9 FIA Q80/S16; only first query per owner consumed'"
    if source.count(before) != 1:
        raise RuntimeError('ROOT235 exact independent reference backend descriptor')
    source = source.replace(before, after)
    nodes = [copy.deepcopy(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'run']
    namespace = dict(vars(old), target_step=target_step)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 __file__+'::original-independent-history-loop', 'exec'), namespace)
    return namespace['run'](*args, **kwargs)
