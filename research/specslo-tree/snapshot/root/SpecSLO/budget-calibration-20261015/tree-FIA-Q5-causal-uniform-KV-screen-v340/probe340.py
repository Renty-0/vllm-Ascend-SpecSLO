"""Finite synthetic attention screen; no model or decoding qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

PREFIXES = (0, 97, 123, 127, 128, 511, 1505, 2555)
HEADS = ((24, 3), (16, 2))
GROUPS = 24
WIDTH = 5
REPEATS = 64
ROUNDS = 20


def variants(prefix):
    minimum = ((prefix+WIDTH+511)//512)*512
    return [('FULL'+str(extent), extent, None) for extent in range(minimum, 2561, 512)] + [
        ('causal-Native', None, None), ('causal-inner1', None, 1)]


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
    torch.set_num_threads(1)
    torch.npu.set_device(1)
    torch.npu.config.allow_internal_format = True
    device = torch.device('npu:1')
    generator = torch.Generator(device='cpu').manual_seed(340)
    records, timings = [], []
    for heads, kvheads in HEADS:
        shape = (544, 128, kvheads*128)
        key = torch.randn(shape, dtype=torch.bfloat16, generator=generator).to(device)
        value = torch.randn(shape, dtype=torch.bfloat16, generator=generator).to(device)
        tables = torch.tensor([list(range(20))+[543]*12]*GROUPS, dtype=torch.int32, device=device)
        causal_mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8, device=device), diagonal=1)
        base = torch.randn((WIDTH, heads, 128), dtype=torch.bfloat16, generator=generator)
        query = base.repeat(GROUPS, 1, 1).to(device)
        mask_host = torch.ones((GROUPS, 1, 10, 4096), dtype=torch.bool)
        mask = mask_host.to(device)

        def parameters(prefix, extent, precise, depth=None):
            mask_host.fill_(True)
            if depth is None:
                for row in range(WIDTH):
                    mask_host[:, :, row, :prefix+row+1] = False
            else:
                mask_host[:, :, 0, :prefix+depth+1] = False
                mask_host[:, :, 1:WIDTH, 0] = False
            mask.copy_(mask_host)
            fields = dict(query=query, key=key, value=value, block_table=tables,
                input_layout='TND', block_size=128,
                actual_seq_lengths=list(range(WIDTH, WIDTH*GROUPS+1, WIDTH)),
                actual_seq_lengths_kv=[extent if extent is not None else prefix+(depth or 0)+WIDTH]*GROUPS,
                num_key_value_heads=kvheads, num_heads=heads, scale=128**-.5)
            if extent is None:
                fields.update(atten_mask=causal_mask, sparse_mode=3, next_tokens=0)
                if precise is not None:
                    fields['inner_precise'] = precise
            else:
                fields.update(atten_mask=mask, sparse_mode=1, inner_precise=1)
            return fields

        def execute(fields, out=None, lse=None):
            out = torch.empty_like(query) if out is None else out
            lse = torch.empty(1, dtype=query.dtype, device=device) if lse is None else lse
            torch_npu.npu_fused_infer_attention_score.out(**fields, out=[out, lse])
            return out

        for prefix in PREFIXES:
            query.copy_(base.repeat(GROUPS, 1, 1))
            for name, extent, precise in variants(prefix):
                actual_original = None
                for phase in ('A1', 'B', 'A2'):
                    actual_host = base.repeat(GROUPS, 1, 1)
                    # A/B/A checks deterministic restoration for a changed
                    # complete query tensor. Each reference keeps the selected
                    # Q identical while changing its other masked queries.
                    if phase == 'B':
                        actual_host += torch.tensor(.125, dtype=torch.bfloat16)
                    query.copy_(actual_host)
                    actual = execute(parameters(prefix, extent, precise)).cpu()
                    common = execute(parameters(prefix, 2560, None)).cpu()
                    reference_equal, reference_error, group_equal = [], [], []
                    for depth in range(WIDTH):
                        host = actual_host.clone()
                        host[::WIDTH] = actual_host[depth::WIDTH]
                        if phase == 'B':
                            for j in range(1, WIDTH):
                                host[j::WIDTH] += torch.tensor(.25, dtype=torch.bfloat16)
                        query.copy_(host)
                        own = execute(parameters(prefix, extent, precise, depth)).cpu()[::WIDTH]
                        selected = actual[depth::WIDTH]
                        reference_equal.append(torch.equal(own.view(torch.int16), selected.view(torch.int16)))
                        reference_error.append(float((own.float()-selected.float()).abs().max()))
                        group_equal.append(torch.equal(selected.view(torch.int16), selected[:1].expand_as(selected).view(torch.int16)))
                    restored = None
                    if phase == 'A1':
                        actual_original = actual.clone()
                    elif phase == 'A2':
                        restored = torch.equal(actual.view(torch.int16), actual_original.view(torch.int16))
                        assert restored
                    records.append(dict(heads=heads, KVheads=kvheads, prefix=prefix,
                        name=name, extent=extent, inner_precise=precise, phase=phase,
                        physical_M=120, groups=GROUPS, width=WIDTH,
                        independent_root_raw16_equal_by_depth=reference_equal,
                        independent_root_max_abs_by_depth=reference_error,
                        group_position_raw16_equal_by_depth=group_equal,
                        all_outputs_equal_uniform_FULL2560=torch.equal(actual.view(torch.int16), common.view(torch.int16)),
                        max_abs_uniform_FULL2560=float((actual.float()-common.float()).abs().max()),
                        complete_output_restored=restored, finite=bool(torch.isfinite(actual).all())))
                print(json.dumps(dict(heads=heads,prefix=prefix,name=name,records=len(records))),flush=True)

            # Static graph costs are operator costs only. Runtime metadata
            # preparation, task refresh, full model and decoding are excluded.
            shortest = variants(prefix)[0]
            choices = [('FULL2560',2560,None)] + ([] if shortest[0]=='FULL2560' else [shortest]) + [
                ('causal-Native',None,None),('causal-inner1',None,1)]
            query.copy_(base.repeat(GROUPS,1,1))
            arms = {}
            for name, extent, precise in choices:
                fields = parameters(prefix,extent,precise)
                # Every graph owns an immutable mask copy.
                fields['atten_mask'] = fields['atten_mask'].clone()
                outputs = [torch.empty_like(query) for _ in range(REPEATS)]
                lse = torch.empty(1,dtype=query.dtype,device=device)
                expected = execute(fields).cpu()
                for _ in range(3):execute(fields,outputs[0],lse)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    for out in outputs:execute(fields,out,lse)
                graph.replay();torch.npu.synchronize()
                assert all(torch.equal(out.cpu().view(torch.int16),expected.view(torch.int16)) for out in outputs)
                arms[name] = dict(graph=graph,outputs=outputs,mask=fields['atten_mask'],lse=lse)
            names = list(arms)
            pairs = []
            for rotation in range(ROUNDS):
                order = names[rotation%len(names):]+names[:rotation%len(names)]
                milliseconds = {}
                for name in order:
                    torch.npu.synchronize();started=time.perf_counter()
                    arms[name]['graph'].replay();torch.npu.synchronize()
                    milliseconds[name]=(time.perf_counter()-started)*1000/REPEATS
                pairs.append(dict(order=order,milliseconds_per_FIA=milliseconds))
            timings.append(dict(heads=heads,KVheads=kvheads,prefix=prefix,
                names=names,repeats=REPEATS,rotations=ROUNDS,pairs=pairs,
                all_graph_outputs_raw16_equal_eager=True,
                median_ms={name:statistics.median(p['milliseconds_per_FIA'][name] for p in pairs) for name in names}))
            del arms
            torch.npu.synchronize()
        del key,value
        torch.npu.empty_cache()
    result = dict(schema='ROOT340.synthetic-Q5-causal-uniform-KV.v1',seed=340,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=records,timings=timings,models_loaded=False,model_qualified=False,
        full_model_or_service_performance=False,Goodput=None,
        timing_scope='static FIA graphs only;64 calls per graph,20 rotated rounds; no runtime preparation or task update')
    with args.output.open('x') as saved:
        json.dump(result,saved,indent=2);saved.write('\n')
    print(json.dumps(dict(status='complete',rows=len(records),timings=len(timings))),flush=True)


if __name__ == '__main__':
    main()
