"""Fuse the exact PA frontier argmax and existing FastV5 top2 chain.

The original FastV5 confidence encoding is retained. This does not change
model weights or model precision. Native captures/checks remain authoritative.
"""

SCALE = 1_000_000


def execute(model, input_ids, positions, attention_metadatas, vocabulary_size, *,
            final_kv_only=False, logical_row_count=None):
    import torch
    if final_kv_only or not 2 <= len(positions) == len(attention_metadatas) <= 5:
        raise ValueError('ROOT373 one frontier plus1to4 original primary steps')
    outputs = []
    current = input_ids
    for step, (position, metadata) in enumerate(zip(positions, attention_metadatas, strict=True)):
        hidden = model(current, position, metadata)
        if step == 0:
            # Match the old run_target_greedy frontier, including tie behavior.
            predicted = model.compute_greedy_tokens(hidden, vocabulary_size)
            outputs.append(torch.stack((predicted * (SCALE + 1), torch.zeros_like(predicted)), dim=-1))
            # Old code slices the real frontier roots on CPU, then pads the
            # next chain's initial IDs with zero. Derive the same mask from
            # Native's copied slot mapping, rather than capture a Python N.
            current = torch.where(metadata.slot_mapping >= 0, predicted, torch.zeros_like(predicted))
            continue
        logits = model.compute_logits(hidden)[:, :vocabulary_size]
        probabilities = torch.softmax(logits.float(), dim=-1)
        confidence, tokens = probabilities.topk(2, dim=-1)
        current = tokens[:, 0]
        encoded = tokens.to(torch.long) * (SCALE + 1) + torch.round(
            confidence.clamp(0, 1) * SCALE).to(torch.long)
        outputs.append(encoded)
    return torch.stack(outputs, dim=1)
