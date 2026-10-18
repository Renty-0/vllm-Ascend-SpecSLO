"""Capacity64 source-only overlay of V85 startup; no actual qualification."""


def make_config():
    from vllm_ascend.spec_decode.pearl.api import PEARLConfig
    return PEARLConfig(
        draft_model_path='/root/data/shared-models/Qwen3-0.6B',
        target_model_path='/root/data/shared-models/Qwen3-32B',
        draft_tensor_parallel_size=1, target_tensor_parallel_size=3,
        draft_dtype='auto', target_dtype='auto', draft_mode='serial_linear',
        max_num_batched_tokens=4096*32, max_num_seqs=64, max_num_queued_seqs=463,
        max_model_len=4096, gpu_memory_utilization=.8,
        num_kvcache_blocks=544, max_aclgraph_entries=512,
        target_verification_graph_buckets=8,
        enable_prefix_caching=False, enable_continuous_batching=True,
        enable_preemptive_scheduling=True, enable_spec_rhythm=True,
        spec_rhythm_linear_full_window=False, spec_rhythm_prefill_token_chunk_size=0,
        spec_rhythm_online_prefill=True, spec_rhythm_merge_ready_homes=True,
        spec_rhythm_prefill_coalesce_min_requests=1,
        spec_rhythm_prefill_coalesce_max_wait_ms=0,
        spec_rhythm_stable_graphs=True, spec_rhythm_priority_burst=2,
        spec_rhythm_tree_width=2, spec_rhythm_tree_depth=4,
        spec_rhythm_min_gamma=1, spec_rhythm_max_eager_tokens=5,
        spec_rhythm_max_target_batch=32, spec_rhythm_target_fallback_max_batch=4,
        spec_rhythm_acceptance_floor=.4, spec_rhythm_verification_budget=96,
        spec_rhythm_cpu_verdict=True,
        draft_use_paged_attention=True, target_use_paged_attention=True,
        enable_mc2=True,
        mc2_profile='/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/mc2-profile.json',
        gamma=8, seed=0, worker_timeout_seconds=6000,
        compact_worker_result_metrics=True)
