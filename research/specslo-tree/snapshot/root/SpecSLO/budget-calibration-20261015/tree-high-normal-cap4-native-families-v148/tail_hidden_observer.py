"""Retain existing captured tail outputs, without inserting device operations.

References extend tensor lifetimes and are part of this candidate's pinned
implementation, including cost measurements. They cannot be removed while
claiming this candidate's qualification. Native returned outputs stay encoded.
"""


def install(native_graph):
    import torch
    import fast_tree_draft_v5 as fast
    original = fast.execute_top2

    def observed(runner, inputs, positions, metadata, vocabulary_size,
                 *, final_kv_only=False, return_confidence=False):
        outputs, hidden_rows = [], []
        for step, (pos, attn) in enumerate(zip(positions, metadata)):
            hidden = runner.model(inputs, pos, attn)
            hidden_rows.append(hidden)
            if final_kv_only and step == len(positions)-1:
                continue
            logits = runner.model.compute_logits(hidden)[:, :vocabulary_size]
            probabilities = torch.softmax(logits.float(), dim=-1)
            confidence, tokens = probabilities.topk(2, dim=-1)
            inputs = tokens[:, 0]
            outputs.append(tokens.long()*(fast.SCALE+1)+torch.round(
                confidence.clamp(0, 1)*fast.SCALE).long())
        key = getattr(runner, 'last_draft_entry_key', None)
        if key is None:
            # CPU model tests have no Native entry dispatcher. Real capture
            # always supplies the complete authoritative key, including lanes.
            key = (f'draft-greedy:{vocabulary_size}|steps:{len(positions)}|paged|confidence',
                   int(positions[0].numel()))
        if native_graph._CAPTURED_TASKS is not None:
            saved = getattr(runner, '_v35_captured_tail_hidden', None)
            if saved is None: saved = runner._v35_captured_tail_hidden = {}
            saved[key] = tuple(hidden_rows)
        else:
            runner._v35_last_eager_tail_hidden = tuple(hidden_rows)
            saved = getattr(runner, '_v35_eager_tail_hidden_by_key', None)
            if saved is None: saved = runner._v35_eager_tail_hidden_by_key = {}
            saved[key] = tuple(hidden_rows)
        return torch.stack(outputs, dim=1)

    fast.execute_top2 = observed
    return original


def snapshot(runner, key, steps):
    saved = getattr(runner, '_v35_captured_tail_hidden', {}).get(key)
    if saved is None or len(saved) != steps:
        raise RuntimeError('No original captured per-step hidden buffers for this exact tail key')
    intervals = []
    for hidden in saved:
        if hidden.ndim != 2 or hidden.shape[0] != key[1] or not hidden.is_contiguous():
            raise RuntimeError('Captured tail hidden shape/layout cannot support all-hidden qualification')
        start = hidden.data_ptr(); end = start+hidden.numel()*hidden.element_size()
        if any(start < other_end and other_start < end for other_start,other_end in intervals):
            raise RuntimeError('Captured tail hidden buffers alias; per-step evidence is unavailable')
        intervals.append((start,end))
    eager = getattr(runner, '_v35_eager_tail_hidden_by_key', {}).get(key)
    if eager is None or len(eager) != steps:
        raise RuntimeError('Same-key original eager-validation hidden buffers are unavailable')
    eager_intervals = []
    for hidden in eager:
        if hidden.ndim != 2 or hidden.shape[0] != key[1] or not hidden.is_contiguous():
            raise RuntimeError('Same-key eager hidden layout cannot prove capture storage isolation')
        start = hidden.data_ptr();end = start+hidden.numel()*hidden.element_size()
        if any(start < capture_end and capture_start < end for capture_start,capture_end in intervals):
            raise RuntimeError('Captured hidden storage aliases same-key eager-validation hidden')
        eager_intervals.append((start,end))
    return tuple(hidden.clone() for hidden in saved),tuple(intervals),tuple(eager_intervals)
