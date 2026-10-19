"""Teacher-forced committed-suffix repair using native one-query PA graphs.

Keeps exactly the same attention path as sequential PA. This experimental
reference favors deterministic contextual KV over the multi-query FIA repair.
"""


def run(engine, repairs, native):
    import torch
    total = padding = calls = 0
    for step in range(max(len(tokens) for _, _, tokens, _ in repairs)):
        active = [(seq, start + step, tokens[step])
                  for seq, start, tokens, _ in repairs if step < len(tokens)]
        ids, positions, tokens = map(list, zip(*active))
        inputs = torch.tensor(tokens, dtype=torch.long, device=engine.device)
        pos, metadata = engine._prepare_attention_metadata(ids, positions, False)
        bucket = native._next_linear_draft_graph_bucket(len(active), engine.config.max_num_seqs)
        inputs, pos, metadata = native.NativeACLGraphRunner._pad_inputs(inputs, pos, metadata, bucket)
        engine.graph_runner.run_tree_hidden(inputs, pos, metadata)
        total += int(inputs.numel())
        padding += int(inputs.numel()) - len(active)
        calls += 1
    return total, padding, calls
