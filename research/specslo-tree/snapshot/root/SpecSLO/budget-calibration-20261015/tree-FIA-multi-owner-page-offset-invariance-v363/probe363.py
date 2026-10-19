"""Synthetic FIA page-offset screen; no model weights or serving measurement."""
import argparse
import hashlib
import json
from pathlib import Path


PATTERNS = ((2, 3), (3, 2), (2, 2, 1), (1, 4), (4, 1), (1, 1, 1, 1, 1), (2, 2))


def configurations():
    result = []
    for sizes in PATTERNS:
        profiles = [tuple([prefix] * len(sizes)) for prefix in (0, 127, 128, 511)]
        if len(sizes) == 2:
            profiles += [(2555, 0), (0, 2555), (2555, 1505), (1505, 2555)]
        else:
            profiles += [(2555,) + (0,) * (len(sizes) - 1), (0,) * (len(sizes) - 1) + (2555,)]
        for prefixes in profiles:
            pages = tuple((prefix + size + 127) // 128 for prefix, size in zip(prefixes, sizes, strict=True))
            for extent in (2560, 4096):
                if sum(pages) * 128 > extent:
                    continue
                for heads in (2, 3):
                    for capacity in (10, 11):
                        for precise in (1, 2):
                            result.append(dict(sizes=sizes, prefixes=prefixes, pages=pages,
                                               extent=extent, KV_heads=heads, Q_heads=8 * heads,
                                               mask_capacity=capacity, inner_precise=precise))
    return tuple(result)


def description(case):
    sizes, prefixes, pages = case['sizes'], case['prefixes'], case['pages']
    if (case['KV_heads'] not in (2, 3) or case['Q_heads'] != 8 * case['KV_heads']
            or case['mask_capacity'] not in (10, 11) or case['inner_precise'] not in (1, 2)):
        raise ValueError('actual Native TP3 head shapes and explicit original precision/mask modes')
    if not 1 <= len(sizes) <= 5 or sum(sizes) > 5 or case['extent'] not in (2560, 4096):
        raise ValueError('bounded declared Q5 segment composition')
    offsets = []
    cursor = 0
    for prefix, size, page_count in zip(prefixes, sizes, pages, strict=True):
        if not 0 <= prefix or not 1 <= size <= 5 or prefix + size > 2560:
            raise ValueError('every original reference query has complete FULL2560 visibility')
        if page_count != (prefix + size + 127) // 128:
            raise ValueError('complete whole-page source extent')
        offsets.append(cursor * 128)
        cursor += page_count
    if cursor * 128 > case['extent']:
        raise ValueError('combined virtual pages must fit declared extent')
    owners = [owner for owner, size in enumerate(sizes) for _ in range(size)]
    depths = [depth for size in sizes for depth in range(size)]
    table = [page for owner, count in enumerate(pages) for page in range(owner * 32, owner * 32 + count)]
    table += [543] * (32 - len(table))
    reference_tables = [list(range(owner * 32, owner * 32 + count)) + [543] * (32 - count)
                        for owner, count in enumerate(pages)]
    return dict(offsets=offsets, owners=owners, depths=depths, packed_table=table,
                reference_tables=reference_tables, compared_local_queries=list(range(sum(sizes))))


def host_masks(case):
    import torch
    meta = description(case)
    shape = (24, 1, case['mask_capacity'], 4096)
    packed = torch.ones(shape, dtype=torch.bool)
    references = [torch.ones(shape, dtype=torch.bool) for _ in case['sizes']]
    for query in range(5):
        if query >= len(meta['owners']):
            packed[:, :, query, 0] = False
        for owner, mask in enumerate(references):
            if query >= len(meta['owners']) or meta['owners'][query] != owner:
                mask[:, :, query, 0] = False
    for query, (owner, depth) in enumerate(zip(meta['owners'], meta['depths'], strict=True)):
        visible = case['prefixes'][owner] + depth + 1
        start = meta['offsets'][owner]
        packed[:, :, query, start:start + visible] = False
        references[owner][:, :, query, :visible] = False
    return packed, references, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--logical-device', type=int, default=1)
    args = parser.parse_args()
    assert not args.output.exists() and args.logical_device == 1
    import os
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')
    if len(visible.split(',')) != 4 or len(set(visible.split(','))) != 4:
        raise RuntimeError('ROOT363 explicit four-card runtime mapping required')
    import torch
    import torch_npu
    torch.set_num_threads(1)
    torch.npu.set_device(args.logical_device)
    torch.npu.config.allow_internal_format = True
    device = torch.device(f'npu:{args.logical_device}')
    generator = torch.Generator(device='cpu').manual_seed(363)
    caches = {}
    for heads in (2, 3):
        shape = (544, 128, heads, 128)
        key = torch.randn(shape, dtype=torch.bfloat16, generator=generator).to(device)
        value = torch.randn(shape, dtype=torch.bfloat16, generator=generator).to(device)
        key[543].zero_()
        value[543].zero_()
        query = torch.randn((120, 8 * heads, 128), dtype=torch.bfloat16, generator=generator)
        caches[heads] = key, value, query
    records = []
    checkpoints = args.output.with_name('completed-cases363.jsonl')
    with checkpoints.open('x') as saved_cases:
        for ordinal, case in enumerate(configurations()):
            key, value, original_query = caches[case['KV_heads']]
            packed_host, reference_host, metadata = host_masks(case)
            packed_mask = packed_host.to(device)
            reference_masks = [mask.to(device) for mask in reference_host]
            packed_table = torch.tensor([metadata['packed_table']] * 24, dtype=torch.int32, device=device)
            reference_tables = [torch.tensor([table] * 24, dtype=torch.int32, device=device)
                                for table in metadata['reference_tables']]
            selected_host = [group * 5 + query for group in range(24) for query in metadata['compared_local_queries']]
            selected = torch.tensor(selected_host, dtype=torch.long)
            restored = None
            baseline = None
            for phase, changed in (('A1', False), ('B', True), ('A2', False)):
                query = (original_query + torch.tensor(.125, dtype=torch.bfloat16) if changed else original_query).to(device)

                def execute(table, mask, extent):
                    output = torch.empty_like(query)
                    lse = torch.empty(1, dtype=query.dtype, device=device)
                    torch_npu.npu_fused_infer_attention_score.out(
                        query=query, key=key.view(544, 128, -1), value=value.view(544, 128, -1),
                        atten_mask=mask, block_table=table, input_layout='TND', block_size=128,
                        actual_seq_lengths=list(range(5, 121, 5)), actual_seq_lengths_kv=[extent] * 24,
                        num_key_value_heads=case['KV_heads'], num_heads=case['Q_heads'], scale=128 ** -.5,
                        sparse_mode=1, inner_precise=case['inner_precise'], out=[output, lse])
                    torch.npu.synchronize()
                    result = output.cpu()
                    if not bool(torch.isfinite(result).all()):
                        raise RuntimeError('ROOT363 nonfinite synthetic FIA result')
                    return result

                expected = torch.empty_like(original_query)
                same_extent = torch.empty_like(original_query)
                for owner, (table, mask) in enumerate(zip(reference_tables, reference_masks, strict=True)):
                    actual_rows = torch.tensor([group * 5 + query for group in range(24)
                                                for query, source in enumerate(metadata['owners']) if source == owner])
                    reference = execute(table, mask, 2560)
                    extended = execute(table, mask, case['extent']) if case['extent'] != 2560 else reference
                    expected.index_copy_(0, actual_rows, reference.index_select(0, actual_rows))
                    same_extent.index_copy_(0, actual_rows, extended.index_select(0, actual_rows))
                packed = execute(packed_table, packed_mask, case['extent'])
                if phase == 'A1':
                    baseline = packed.clone()
                elif phase == 'A2':
                    restored = torch.equal(packed.view(torch.int16), baseline.view(torch.int16))
                    if not restored:
                        raise RuntimeError('ROOT363 full packed output failed ABA restoration')
                compared = packed.index_select(0, selected)
                reference = expected.index_select(0, selected)
                extended = same_extent.index_select(0, selected)
                raw_equal = (compared.view(torch.int16) == reference.view(torch.int16)).flatten(1).all(1)
                offset_equal = (compared.view(torch.int16) == extended.view(torch.int16)).flatten(1).all(1)
                extent_equal = (extended.view(torch.int16) == reference.view(torch.int16)).flatten(1).all(1)
                difference = (compared.float() - reference.float()).abs().flatten(1).amax(1)
                row = dict(case_ordinal=ordinal, case=case, phase=phase,
                           compared_queries=len(selected_host), physical_M=120, groups=24, width=5,
                           all_compared_raw16_equal=bool(raw_equal.all()),
                           same_extent_page_offset_raw16_equal=bool(offset_equal.all()),
                           unshifted_extent_vs_FULL2560_raw16_equal=bool(extent_equal.all()),
                           different_query_indices=(~raw_equal).nonzero().flatten().tolist(),
                           max_abs_by_query=difference.tolist(), whole_output_restored=restored,
                           identical_ordered_real_KV_and_original_query=True)
                records.append(row)
                saved_cases.write(json.dumps(row) + '\n')
                saved_cases.flush()
            print(json.dumps(dict(case_ordinal=ordinal, complete_cases=ordinal + 1,
                                  raw16_equal_all_phases=all(row['all_compared_raw16_equal'] for row in records[-3:]),
                                  models_loaded=False)), flush=True)
    result = dict(schema='ROOT363.synthetic-FIA-multi-owner-page-offset.v1', seed=363, rows=records,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  complete_cases=len(configurations()), visible_runtime_ids=visible,
                  logical_device=args.logical_device, physical_cards_requested=[2, 5, 6, 7],
                  models_loaded=False, model_qualified=False,
                  performance=False, Raw=None, Goodput=None, adopted=False)
    with args.output.open('x') as saved:
        json.dump(result, saved, indent=2)
        saved.write('\n')
    print(json.dumps(dict(status='complete', cases=len(configurations()), rows=len(records),
                          noninvariant_rows=sum(not row['all_compared_raw16_equal'] for row in records),
                          models_loaded=False, performance=False)), flush=True)


if __name__ == '__main__':
    main()
