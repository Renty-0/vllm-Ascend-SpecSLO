"""Synthetic group/query invariance; no model, decoding or service metrics."""
import argparse
import hashlib
import json
from pathlib import Path


def configurations():
    return tuple((width, groups, mode, capacity, precise, prefix)
        for width in (4, 5)
        for groups in (24, 32, 33)
        for mode in ('roots', 'all_queries')
        for capacity in (10, 11)
        for precise in (1, 2)
        for prefix in (0, 97, 127, 128, 1505, 2555))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--logical-device', type=int, default=1)
    args = parser.parse_args()
    assert not args.output.exists() and args.logical_device == 1
    import os
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    import torch
    import torch_npu

    torch.npu.set_device(args.logical_device)
    torch.npu.config.allow_internal_format = True
    device = torch.device(f'npu:{args.logical_device}')
    generator = torch.Generator(device='cpu').manual_seed(335)
    shape = (544, 128, 3, 128)
    key = torch.randn(shape, dtype=torch.bfloat16, generator=generator).to(device)
    value = torch.randn(shape, dtype=torch.bfloat16, generator=generator).to(device)
    records = []
    for width in (4, 5):
        for groups in (24, 32, 33):
            original = torch.randn((width*groups, 24, 128), dtype=torch.bfloat16,
                                   generator=generator)
            root = original[0].clone()
            table = torch.tensor([list(range(20))+[543]*12]*groups,
                                 dtype=torch.int32, device=device)
            for mode in ('roots', 'all_queries'):
                query_host = original.clone()
                selected = slice(None, None, width) if mode == 'roots' else slice(None)
                query_host[selected].copy_(root)
                query = query_host.to(device)
                for capacity in (10, 11):
                    mask_host = torch.ones((groups, 1, capacity, 4096), dtype=torch.bool)
                    if mode == 'roots':
                        mask_host[:, :, 1:width, 0] = False
                    mask = mask_host.to(device)
                    for precise in (1, 2):
                        for prefix in (0, 97, 127, 128, 1505, 2555):
                            visible = slice(0, 1) if mode == 'roots' else slice(0, width)
                            mask_host[:, :, visible, :] = True
                            mask_host[:, :, visible, :prefix+1] = False
                            mask.copy_(mask_host)
                            baseline = None
                            for phase, changed in (('A1', False), ('B', True), ('A2', False)):
                                payload = query_host.clone()
                                if changed:
                                    payload[selected].copy_(root+torch.tensor(.125, dtype=torch.bfloat16))
                                query.copy_(payload)
                                out = torch.empty_like(query)
                                lse = torch.empty(1, dtype=query.dtype, device=device)
                                torch_npu.npu_fused_infer_attention_score.out(
                                    query=query, key=key.view(544, 128, -1),
                                    value=value.view(544, 128, -1),
                                    atten_mask=mask, block_table=table, input_layout='TND',
                                    block_size=128,
                                    actual_seq_lengths=list(range(width, width*groups+1, width)),
                                    actual_seq_lengths_kv=[2560]*groups,
                                    num_key_value_heads=3, num_heads=24, scale=128**-.5,
                                    sparse_mode=1, inner_precise=precise, out=[out, lse])
                                torch.npu.synchronize()
                                result = out.cpu()
                                assert bool(torch.isfinite(result).all())
                                compared = result[selected]
                                difference = (compared.float()-compared[0].float()).abs().flatten(1).amax(1)
                                equal = (compared.view(torch.int16) == compared[0].view(torch.int16)).flatten(1).all(1)
                                restored = None
                                if phase == 'A1':
                                    baseline = result.clone()
                                elif phase == 'A2':
                                    restored = torch.equal(result.view(torch.int16), baseline.view(torch.int16))
                                    assert restored, 'same input must restore the complete raw16 output'
                                row = dict(width=width, groups=groups, M=width*groups, mode=mode,
                                    mask_capacity=capacity, inner_precise=precise, prefix=prefix, phase=phase,
                                    compared_queries=compared.shape[0],
                                    identical_Q_K_V_visibility_and_page_table=True,
                                    all_compared_raw16_equal=bool(equal.all()),
                                    different_query_indices=(~equal).nonzero().flatten().tolist(),
                                    max_abs_by_query=difference.tolist(), whole_output_restored=restored)
                                records.append(row)
                                print(json.dumps({k:v for k,v in row.items()
                                    if k not in ('different_query_indices', 'max_abs_by_query')}), flush=True)
    assert len(records) == len(configurations())*3 == 864
    result = dict(schema='ROOT335.synthetic-FIA-group-query-invariance.v1', rows=records,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), seed=335,
        models_loaded=False, model_qualified=False, performance=False, Goodput=None)
    with args.output.open('x') as saved:
        json.dump(result, saved, indent=2)
        saved.write('\n')
    print(json.dumps(dict(status='complete', rows=len(records),
        noninvariant_rows=sum(not r['all_compared_raw16_equal'] for r in records),
        models_loaded=False, performance=False)), flush=True)


if __name__ == '__main__':
    main()
