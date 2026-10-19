"""HOST candidate for three dependent remaining primary steps with their original top2.

The two tokens and two float32 probabilities are stored as two int64 words
per step using dtype views. This is lossless byte packing, with no probability
rounding or model quantization. No runner or serving method is installed here.
"""


def execute(model, input_ids, positions, attention_metadatas, vocabulary_size, *, final_kv_only=False,
            logical_row_count=None):
    import torch
    steps = 4 if final_kv_only else 3
    if len(positions) != steps or len(attention_metadatas) != steps:
        raise ValueError('explicit three remaining primary steps and original step metadata')
    outputs = []
    current = input_ids
    logical_row_count = input_ids.numel() if logical_row_count is None else logical_row_count
    if type(logical_row_count) is not int or not 1 <= logical_row_count <= input_ids.numel():
        raise ValueError('explicit real request prefix within original physical input rows')
    real = torch.arange(input_ids.numel(), device=input_ids.device) < logical_row_count
    for step, (pos, metadata) in enumerate(zip(positions, attention_metadatas, strict=True)):
        hidden = model(current, pos, metadata)
        if final_kv_only and step == 3:
            continue
        logits = model.compute_logits(hidden)[:, :vocabulary_size]
        # Match the original greedy draft_tree_forward: float32 softmax,
        # then probabilities.topk(2), with primary token feeding the next step.
        probabilities = torch.softmax(logits.float(), dim=-1)
        values, tokens = probabilities.topk(2, dim=-1)
        # The original tree padder reuses fixed dummy roots at every depth.
        # Only real requests follow the newly predicted primary token.
        current = torch.where(real, tokens[:, 0], input_ids).contiguous()
        token_word = tokens.to(torch.int32).contiguous().view(torch.int64).squeeze(-1)
        probability_word = values.contiguous().view(torch.int64).squeeze(-1)
        outputs.append(torch.stack((token_word, probability_word), dim=-1))
    return torch.stack(outputs, dim=1)


def decode(packed, vocabulary_size):
    import torch
    if (packed.device.type != 'cpu' or packed.dtype != torch.long or packed.ndim != 3
            or tuple(packed.shape[1:]) != (3, 2)):
        raise ValueError('explicit HOST int64[N,3,2] lossless graph output')
    n = packed.shape[0]
    tokens = packed[:, :, 0].contiguous().view(torch.int32).reshape(n, 3, 2)
    probabilities = packed[:, :, 1].contiguous().view(torch.float32).reshape(n, 3, 2)
    if not bool(((tokens >= 0) & (tokens < vocabulary_size)).all()):
        raise ValueError('all original top2 token IDs must fit the vocabulary')
    if not bool((torch.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)).all()):
        raise ValueError('all original float32 probabilities must remain finite and bounded')
    raw_tokens = torch.cat((tokens[:, :, 0], tokens[:, :, 1]), dim=1)
    raw_probabilities = torch.cat((probabilities[:, :, 0], probabilities[:, :, 1]), dim=1)
    return raw_tokens, raw_probabilities
