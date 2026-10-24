"""Real-device raw16 eager/graph precheck, independent of model arithmetic."""


def mappings():
    result = []
    for counts in ((), (1,), (127,), (128,), (132,), (127, 128, 132, 1), (128,) * 12):
        source, dest = [], []
        for i, count in enumerate(counts):
            for j in range(count):
                # Nonadjacent canonical pages force real segment boundaries.
                src = i * 256 + j if j < 124 else 4096 + i * 256 + j
                source.append(src)
                dest.append(8192 + i * 256 + j)
        for j in range(1792 - len(source)):
            source.append(12288 + j)
            dest.append(12288 + j)
        result.append((source, dest))
    return result


def run(engine):
    import torch
    from span_kernel import copy_spans
    from spans import pack
    width = engine.model.layers[0].self_attn.key_cache.shape[-2] * 128
    if width not in (256, 384):
        raise RuntimeError('Actual current target rank width')
    rows = 16384
    bits = torch.arange(rows * width, dtype=torch.int32, device='cpu').remainder(65536).to(torch.int16)
    base_key = bits.reshape(rows, width).view(torch.bfloat16).to(engine.device)
    base_value = bits.flip(0).reshape(rows, width).view(torch.bfloat16).to(engine.device)
    key, value = base_key.clone(), base_value.clone()
    descriptors = torch.zeros((64, 3), dtype=torch.long, device=engine.device)
    copy_spans(key, value, descriptors)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        copy_spans(key, value, descriptors)
    cases = []
    for index, (source, destination) in enumerate(mappings()):
        compressed = pack(source, destination, rows=rows)
        descriptors.copy_(torch.tensor(compressed['spans'], dtype=torch.long))
        src = torch.tensor(source, dtype=torch.long, device=engine.device)
        dst = torch.tensor(destination, dtype=torch.long, device=engine.device)
        for variant in (0, 1, 0):
            k0, v0 = (base_key, base_value) if variant == 0 else (base_value, base_key)
            expected_key, expected_value = k0.clone(), v0.clone()
            expected_key.index_copy_(0, dst, expected_key.index_select(0, src))
            expected_value.index_copy_(0, dst, expected_value.index_select(0, src))
            for mode in ('eager', 'graph'):
                key.copy_(k0)
                value.copy_(v0)
                if mode == 'eager':
                    copy_spans(key, value, descriptors)
                else:
                    graph.replay()
                torch.npu.synchronize()
                if not (torch.equal(key.view(torch.int16), expected_key.view(torch.int16))
                        and torch.equal(value.view(torch.int16), expected_value.view(torch.int16))):
                    raise RuntimeError('Full raw16 span-copy storage mismatch')
            cases.append(dict(case=index, variant=variant, copied_rows=compressed['copied_rows'],
                              spans=compressed['span_count'], eager_and_graph_full_raw16_exact=True))
    del graph
    torch.npu.synchronize()
    return dict(width=width, cases=cases, raw16_input_patterns=65536,
                initialized_private_rows=rows, same_descriptors_changed_and_restored=True,
                full_storage_compared=True, model_KV_mutated=False)
