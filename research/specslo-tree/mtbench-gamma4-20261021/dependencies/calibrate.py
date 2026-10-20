"""Bounded, paired-context serial gain and compute-envelope calibration.

All model calls use the sealed source. No scheduler grants or production data
are fabricated. This diagnostic deliberately synchronizes outside timed calls.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

W = Path(__file__).resolve().parent
FROZEN = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1')
REFERENCE = Path('/root/SpecSLO/tpot-report-20261015/serial-repeat-once/measurement-output/runs/candidate-on-serial-repeat-20261015-r1/result.json')


def save(name, data):
    p = W / name
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(p)


def config():
    from vllm_ascend.spec_decode.pearl.api import PEARLConfig
    return PEARLConfig(
        draft_model_path='/root/data/shared-models/Qwen3-0.6B',
        target_model_path='/root/data/shared-models/Qwen3-32B',
        draft_tensor_parallel_size=1, target_tensor_parallel_size=3,
        max_num_batched_tokens=4096*64, max_num_seqs=64,
        max_num_queued_seqs=463, max_model_len=4096,
        gpu_memory_utilization=.8, num_kvcache_blocks=544,
        max_aclgraph_entries=128, gamma=4, seed=0,
        enable_prefix_caching=False,
        enable_continuous_batching=True, enable_preemptive_scheduling=True,
        enable_spec_rhythm=True, spec_rhythm_linear_full_window=True,
        spec_rhythm_linear_bonus_token=False,
        spec_rhythm_linear_eager_cross_graph_bucket=True,
        spec_rhythm_linear_idle_residual_eager=True,
        spec_rhythm_online_prefill=True,
        spec_rhythm_prefill_token_chunk_size=512,
        spec_rhythm_merge_ready_homes=True, spec_rhythm_min_gamma=4,
        spec_rhythm_max_eager_tokens=4, spec_rhythm_max_target_batch=32,
        spec_rhythm_target_fallback_max_batch=4,
        spec_rhythm_acceptance_floor=.4, spec_rhythm_priority_burst=2,
        spec_rhythm_tree_width=1, spec_rhythm_tree_depth=1,
        draft_use_paged_attention=True, enable_mc2=True,
        mc2_profile=str(FROZEN/'runtime/mc2-profile.json'),
        worker_timeout_seconds=1800, compact_worker_result_metrics=True)


def prepare():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/root/data/shared-models/Qwen3-0.6B')
    rows = [json.loads(s) for s in (FROZEN/'runtime/manifest.jsonl').read_text().splitlines()]
    result = json.loads(REFERENCE.read_text())['results'][0]
    outputs = result['output_token_ids']
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['category']].append(i)
    selected = []
    rng = random.Random(20261015)
    for category, candidates in sorted(groups.items()):
        rng.shuffle(candidates)
        distinct, seen = [], set()
        for i in candidates:
            key = (rows[i]['source_dataset'], rows[i]['source_index'])
            if key not in seen:
                seen.add(key)
                distinct.append(i)
            if len(distinct) == 24:
                break
        assert len(distinct) == 24
        for j, i in enumerate(distinct):
            row = rows[i]
            prompt = tokenizer.encode(tokenizer.apply_chat_template(
                [{'role':'user','content':row['prompt']}], tokenize=False, add_generation_prompt=True))
            for offset in (32, 96, 192):
                tokens = prompt + outputs[i][:offset]
                assert len(tokens) + 8 <= 4096
                selected.append(dict(request_id=row['request_id'], manifest_index=i,
                    category=category, slo_class=row['slo_class'],
                    source_index=row['source_index'], source_dataset=row['source_dataset'],
                    split='train' if j<16 else 'heldout', output_offset=offset,
                    context_length=len(tokens), token_ids=tokens))
    # Interleave types, split and output positions, using the same contexts at every n.
    rng.shuffle(selected)
    save('inputs.json', dict(reference_path=str(REFERENCE),
        reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        reference_is_context_fixture_not_target_oracle=True,
        manifest_sha256=hashlib.sha256((FROZEN/'runtime/manifest.jsonl').read_bytes()).hexdigest(),
        contexts=selected, requests_per_type=24,
        n_definition='Number of target query positions; full-window, no bonus; n=0 means skipped service',
        n_values=list(range(1,9)), class_models='One Qwen3 pair, three request categories'))
    save('config.json', {k:repr(v) if k=='mc2_profile' else v for k,v in asdict(config()).items()})
    print(json.dumps(dict(contexts=len(selected), categories=list(groups),
                         max_context=max(c['context_length'] for c in selected))))


def allocate_prefill(engine, contexts):
    """Populate only the committed prefix; root and suffix are overwritten per call."""
    engine._release_cache()
    prompts = [c['token_ids'][:-1] for c in contexts]
    engine._allocate_cache(prompts, enable_prefix_caching=False)
    for seq, tokens in enumerate(prompts):
        # Fixed 512-token chunks bound prefill temporaries on long documents.
        for start in range(0, len(tokens), 512):
            chunk = tokens[start:start+512]
            engine._run_packed_hidden(chunk, [seq]*len(chunk), list(range(start,start+len(chunk))),
                use_aclgraph=False, logit_indices=[len(chunk)-1], use_fused_infer_attention=True)
    engine._ensure_cache_capacity(list(range(len(contexts))),
                                  [c['context_length']+8 for c in contexts])


def draft_call(engine, contexts, n):
    import torch
    ids = list(range(len(contexts)))
    first = [c['context_length']-1 for c in contexts]
    pos, metadata = [], []
    for step in range(n):
        p,m = engine._prepare_attention_metadata(ids, [p+step for p in first], False)
        pos.append(p)
        metadata.append(m)
    roots = torch.tensor([c['token_ids'][-1] for c in contexts], dtype=torch.long, device=engine.device)
    return engine.graph_runner.run_draft_greedy(roots, pos, metadata,
        engine.draft_vocab_size).clone()


def target_call(engine, contexts, proposals, n):
    import torch
    roots = torch.tensor([c['token_ids'][-1] for c in contexts], dtype=torch.long,
                         device=engine.device).unsqueeze(1)
    inputs = torch.cat((roots, proposals[:,:n-1]),dim=1).flatten()
    seq = [i for i in range(len(contexts)) for _ in range(n)]
    pos = [c['context_length']-1+j for c in contexts for j in range(n)]
    return engine._run_device_packed_greedy(inputs, seq, pos,
        use_aclgraph=True, use_fused_infer_attention=True).reshape(len(contexts),n).clone()


def snapshot(engine):
    runner = engine.graph_runner
    return dict(captures=runner.capture_count, failed=runner.failed_capture_count,
        shape_fallbacks=runner.shape_fallback_count, capacity_fallbacks=runner.capacity_fallback_count,
        executions=runner.graph_execution_metrics())


def probe(engine, phase):
    import torch
    import torch.distributed as dist
    contexts = json.loads((W/'inputs.json').read_text())['contexts']
    report = dict(rank=engine.rank, phase=phase, diagnostic_only=True, gain=[], timing=[], errors=[])
    def barrier():
        torch.npu.synchronize()
        dist.barrier()
    try:
        if phase == 'gain':
            for batch_index, begin in enumerate(range(0,len(contexts),12)):
                cohort = contexts[begin:begin+12]
                allocate_prefill(engine, cohort)
                barrier()
                proposals = (draft_call(engine,cohort,8) if engine.is_draft else
                    torch.empty((len(cohort),8),dtype=torch.long,device=engine.device))
                dist.broadcast(proposals,src=0)
                proposal_rows = proposals.cpu().tolist()
                # Reverse order on alternate cohorts to spread capture/clock drift.
                for n in (range(1,9) if batch_index%2==0 else range(8,0,-1)):
                    if not engine.is_draft:
                        values = target_call(engine,cohort,proposals,n).cpu().tolist()
                        for c,d,t in zip(cohort,proposal_rows,values):
                            accepted = 0
                            while accepted<n and d[accepted]==t[accepted]:
                                accepted += 1
                            report['gain'].append({k:v for k,v in c.items() if k!='token_ids'} | dict(
                                n=n, accepted=accepted, delivered=min(n,accepted+1),
                                proposals=d[:n], target_tokens=t,
                                committed=d[:accepted]+(t[accepted:accepted+1] if accepted<n else [])))
                barrier()
                save(f'rank{engine.rank}-gain-progress.json',dict(completed_contexts=begin+len(cohort),
                    total_contexts=len(contexts),updated_unix=time.time()))
        elif phase == 'cost':
            ordered = sorted(contexts,key=lambda c:c['context_length'])
            for variant, pool in [('short',ordered[:32]),('long',ordered[-32:])]:
                for m in (1,2,4,8,16,24,32):
                    cohort = pool[:m]
                    allocate_prefill(engine,cohort)
                    barrier()
                    proposals = (draft_call(engine,cohort,8) if engine.is_draft else
                        torch.empty((m,8),dtype=torch.long,device=engine.device))
                    dist.broadcast(proposals,src=0)
                    for n in (1,2,4,6,8):
                        fn = lambda: (draft_call(engine,cohort,n) if engine.is_draft else
                                      target_call(engine,cohort,proposals,n))
                        # Exact graph preparation and validation are excluded from timing.
                        for _ in range(3):
                            fn()
                        torch.npu.synchronize()
                        before = snapshot(engine)
                        for repeat in range(15):
                            barrier()
                            started=time.perf_counter_ns()
                            fn()
                            torch.npu.synchronize()
                            elapsed=(time.perf_counter_ns()-started)/1e6
                            report['timing'].append(dict(variant=variant,requests=m,n=n,B=m*n,
                                repeat=repeat, role='draft' if engine.is_draft else 'target',
                                latency_ms=elapsed,context_min=min(c['context_length'] for c in cohort),
                                context_max=max(c['context_length'] for c in cohort)))
                        after=snapshot(engine)
                        report.setdefault('geometry_checks',[]).append(dict(variant=variant,m=m,n=n,
                            before=before,after=after,measured_capture_delta=after['captures']-before['captures'],
                            measured_fallback_delta=sum(after[k]-before[k] for k in
                                                       ('failed','shape_fallbacks','capacity_fallbacks'))))
                    save(f'rank{engine.rank}-cost-progress.json',dict(variant=variant,requests=m,
                        updated_unix=time.time()))
        else:
            raise ValueError(phase)
        report['final_graph_metrics']=engine.graph_metrics()
        engine._release_cache()
        barrier()
        return report
    except BaseException:
        report['errors'].append(traceback.format_exc())
        raise
    finally:
        save(f'rank{engine.rank}-{phase}.json',report)


def worker(cfg,rank,port,connection,admission_connection=None,compact_worker_result_metrics=False):
    import torch
    import torch.distributed as dist
    from vllm_ascend.spec_decode.pearl.native_engine import NativePearlEngine
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),RANK=str(rank),
        WORLD_SIZE='4',LOCAL_RANK=str(rank))
    if os.environ.get('HCCL_NPU_SOCKET_PORT_RANGE')=='auto':
        os.environ.pop('HCCL_NPU_SOCKET_PORT_RANGE')
    engine=None
    try:
        engine=NativePearlEngine(cfg)
        connection.send(('ready',rank,engine.graph_metrics()))
        while True:
            mode,phase,_,_=connection.recv()
            if mode=='exit':
                break
            assert mode=='calibrate'
            with torch.inference_mode():
                data=probe(engine,phase)
            connection.send(('calibrated',rank,dict(rows=len(data['gain']),timings=len(data['timing']))))
    except BaseException:
        connection.send(('error',traceback.format_exc()))
    finally:
        connection.close()
        if admission_connection is not None:
            admission_connection.close()
        if engine is not None and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--phase',choices=['gain','cost','both'],default='both')
    args=parser.parse_args()
    if args.prepare:
        prepare()
        return
    from vllm_ascend.spec_decode.pearl import api
    api._pearl_worker=worker
    engine=None
    try:
        engine=api.PEARLEngine(config())
        for phase in (['gain','cost'] if args.phase=='both' else [args.phase]):
            print('Starting '+phase,flush=True)
            engine._send_all(('calibrate',phase,None,None))
            replies=engine._receive_all('serial '+phase+' calibration')
            save(phase+'-completed.json',dict(replies=replies,completed_unix=time.time()))
            print(json.dumps(dict(phase=phase,replies=replies)),flush=True)
    finally:
        if engine is not None:
            engine.exit()
        save('worker-exit.json',dict(closed=bool(engine and engine._closed),
            all_exited=bool(engine and all(not p.is_alive() for p in engine._processes))))


if __name__=='__main__':
    main()
