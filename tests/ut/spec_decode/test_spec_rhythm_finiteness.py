# SPDX-License-Identifier: Apache-2.0
"""CPU fault injection for tree/FIA cache and output health boundaries.

The traced-graph test checks device-resident dataflow/storage lifetime on CPU;
it does not substitute for an actual NPU ACLGraph replay regression.
"""

import multiprocessing
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from tests.ut.spec_decode.test_spec_rhythm_tree_loop import _TreeLoopHarness
from vllm_ascend.spec_decode.pearl.native_engine import NativePearlEngine, PearlPipelineState
from vllm_ascend.spec_decode.pearl.native_model import NativeLMHead, NativeQwen2ForCausalLM, NativeTPContext
from vllm_ascend.spec_decode.pearl.roofline import ProfiledRoofline
from vllm_ascend.spec_decode.pearl.topology import PearlTopology
from vllm_ascend.spec_decode.pearl.tree import build_tree_speculation_plan


def _model(track=True):
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        intermediate_size=32,
        tie_word_embeddings=False,
        num_hidden_layers=1,
        pearl_track_cache_finiteness=track,
    )
    model = NativeQwen2ForCausalLM(config, NativeTPContext(None, 0, 1, 0))
    generator = torch.Generator().manual_seed(413)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1, generator=generator)
    model.configure_cache(16)
    return model


def _engine(model=None, *, rank=1):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(enable_spec_rhythm=True, spec_rhythm_tree_width=2, spec_rhythm_tree_depth=2)
    engine.device = torch.device("cpu")
    engine.model = _model() if model is None else model
    engine.rank = rank
    engine.is_draft = rank == 0
    engine.topology = PearlTopology.from_tensor_parallel_sizes(1, 3)
    engine.cache_allocation = SimpleNamespace(num_cached_tokens={0: 0})
    engine.prefix_cache = SimpleNamespace(release=Mock())
    engine.cache_block_tables = torch.zeros(1, 1, dtype=torch.int32)
    return engine


@torch.inference_mode()
def test_tracking_preserves_finite_logits_and_greedy_values_exactly():
    tracked, ordinary = _model(), _model(False)
    tokens, positions = torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2])
    tracked_hidden, ordinary_hidden = tracked(tokens, positions), ordinary(tokens, positions)
    assert torch.equal(tracked_hidden, ordinary_hidden)
    assert torch.equal(tracked.compute_logits(tracked_hidden), ordinary.compute_logits(ordinary_hidden))
    assert torch.equal(
        tracked.compute_greedy_tokens(tracked_hidden, 31), ordinary.compute_greedy_tokens(ordinary_hidden, 31)
    )
    assert not _engine(tracked)._spec_rhythm_nonfinite_flag()
    assert all("nonfinite" not in name for name in tracked.state_dict())


@torch.inference_mode()
def test_ordinary_prefill_metadata_tracks_nonfinite_cache_writes():
    model = _model()
    positions, metadata = model.make_attention_metadata([0, 0], [0, 1])
    assert not metadata.tree_attention
    model.layers[0].self_attn.qkv_proj.weight.fill_(float("nan"))
    model(torch.tensor([1, 2]), positions, metadata)
    assert model.layers[0].self_attn.tree_cache_nonfinite
    assert not torch.isfinite(model.layers[0].self_attn.key_cache).all()


@torch.inference_mode()
def test_tree_decode_fault_is_caught_at_model_boundary_without_layer_reductions():
    model = _model()
    plan = build_tree_speculation_plan(2, 2, 0, 16)
    input_ids, positions, metadata = model.make_tree_attention_metadata(
        [plan],
        [1],
        [[2, 3, 4, 5]],
        model.layers[0].self_attn.block_table,
    )
    handle = model.layers[0].self_attn.qkv_proj.register_forward_hook(
        lambda module, args, output: torch.full_like(output, float("nan"))
    )
    try:
        hidden = model(input_ids, positions, metadata)
        model.compute_greedy_tokens(hidden, 31)
    finally:
        handle.remove()
    assert not model.layers[0].self_attn.tree_cache_nonfinite
    assert model.output_nonfinite
    assert model.lm_head.logits_nonfinite
    assert _engine(model)._spec_rhythm_nonfinite_flag()


@pytest.mark.parametrize("stage", ["mlp", "norm", "head"])
@torch.inference_mode()
def test_final_layer_and_head_faults_cannot_hide_behind_finite_attention(stage):
    model = _model()
    if stage == "mlp":
        handle = model.layers[-1].mlp.register_forward_hook(
            lambda module, args, output: torch.full_like(output, float("nan"))
        )
    elif stage == "norm":
        handle = model.norm.register_forward_hook(
            lambda module, args, output: (torch.full_like(output[0], float("nan")), output[1])
        )
    else:
        handle = None
        model.lm_head.weight.fill_(float("nan"))
    try:
        hidden = model(torch.tensor([1, 2]), torch.tensor([0, 1]))
        tokens = model.compute_greedy_tokens(hidden, 31)
    finally:
        if handle is not None:
            handle.remove()
    assert tokens.dtype == torch.long, "argmax alone does not report nonfinite model outputs"
    assert not model.layers[-1].self_attn.tree_cache_nonfinite
    assert model.lm_head.logits_nonfinite
    assert bool(model.output_nonfinite) == (stage != "head")
    assert _engine(model)._spec_rhythm_nonfinite_flag()


@pytest.mark.parametrize("confidence", [False, True])
@torch.inference_mode()
def test_inactive_tp_vocabulary_negative_infinity_is_not_a_model_fault(monkeypatch, confidence):
    head = NativeLMHead(4, 2, NativeTPContext(None, 1, 2, 0), track_cache_finiteness=True)
    head.weight.fill_(float("nan"))

    def gather(outputs, value, **kwargs):
        for output in outputs:
            output.copy_(value)

    monkeypatch.setattr(dist, "all_gather", gather)
    method = head.greedy_with_confidence if confidence else head.greedy
    method(torch.ones(1, 2), vocabulary_size=2)
    assert not head.logits_nonfinite, "an excluded shard's synthetic -inf is intentional"


@torch.inference_mode()
def test_target_only_vocabulary_suffix_is_excluded_before_greedy_health_check():
    head = NativeLMHead(4, 2, NativeTPContext(None, 0, 1, 0), track_cache_finiteness=True)
    head.weight.fill_(1)
    head.weight[2:].fill_(float("nan"))
    hidden = torch.ones(1, 2)
    head.greedy(hidden, vocabulary_size=2)
    head.greedy_with_confidence(hidden, vocabulary_size=2)
    assert not head.logits_nonfinite
    head(hidden)
    assert head.logits_nonfinite, "full-logits diagnostic exposes and must flag the suffix"


@torch.inference_mode()
def test_sticky_flag_survives_traced_replay_and_release_until_full_cache_reinitialization():
    model = _model()
    head = model.lm_head
    traced = torch.jit.trace(head, torch.ones(1, 16), check_trace=False)
    flag = head.logits_nonfinite
    pointer = flag.data_ptr()
    traced(torch.full((1, 16), float("nan")))
    assert flag
    traced(torch.ones(1, 16))
    assert flag and head.logits_nonfinite.data_ptr() == pointer
    engine = _engine(model)
    attention_flag = model.layers[0].self_attn.tree_cache_nonfinite
    attention_flag.fill_(True)
    engine._release_cache()
    assert flag and attention_flag
    assert engine._spec_rhythm_nonfinite_flag()
    model.configure_cache(16)
    assert head.logits_nonfinite is flag and not flag
    assert model.layers[0].self_attn.tree_cache_nonfinite is attention_flag and not attention_flag
    assert not model.layers[0].self_attn.key_cache.any()
    assert not engine._spec_rhythm_nonfinite_flag()


def test_decode_boundary_omits_layer_flags_already_voted_at_prefill():
    model = _model()
    engine = _engine(model)
    model.layers[0].self_attn.tree_cache_nonfinite.fill_(True)

    assert engine._spec_rhythm_nonfinite_flag()
    assert not engine._spec_rhythm_nonfinite_flag(include_layer_cache=False)
    model.output_nonfinite.fill_(True)
    assert engine._spec_rhythm_nonfinite_flag(include_layer_cache=False)


@pytest.mark.parametrize("bad_rank", [0, 1, 2, 3])
def test_prefill_vote_prevents_max_tokens_one_commit_and_stream(monkeypatch, bad_rank):
    harness = _TreeLoopHarness(monkeypatch, requests=1, capacity=1, max_tokens=1, online_prefill=True)
    engine = harness.engine
    engine.model = _model()
    engine.cache_allocation = SimpleNamespace(num_cached_tokens={0: 0})
    engine._run_packed_sample = Mock(return_value=torch.tensor([7]))
    engine._prefill_and_sample_target_batch = NativePearlEngine._prefill_and_sample_target_batch.__get__(engine)
    if bad_rank == engine.rank:
        engine.model.output_nonfinite.fill_(True)
    votes = []

    def vote(failed, **kwargs):
        assert failed.dtype == torch.int64 and failed.shape == (1,)
        assert kwargs["op"] == dist.ReduceOp.MAX
        votes.append(failed.item())
        failed.fill_(1)

    monkeypatch.setattr(dist, "all_reduce", vote)
    chunks = []
    engine._token_commit_callback = chunks.append
    engine._stream_delivered_counts = {}
    engine._stream_started = 0.0
    with pytest.raises(RuntimeError, match="no first token was committed"):
        harness.run()
    assert votes == [int(bad_rank == engine.rank)]
    assert not chunks
    assert all(not state.committed_completion_token_ids for state in harness.target_states + harness.draft_states)
    assert all(state.delivered_tokens == 0 for state in harness.controllers[0].request_states.values())


def test_finite_prefill_votes_before_one_token_request_completes(monkeypatch):
    harness = _TreeLoopHarness(monkeypatch, requests=1, capacity=1, max_tokens=1, online_prefill=True)
    engine = harness.engine
    engine.model = _model()
    engine.cache_allocation = SimpleNamespace(num_cached_tokens={0: 0})
    engine._run_packed_sample = Mock(return_value=torch.tensor([7]))
    engine._prefill_and_sample_target_batch = NativePearlEngine._prefill_and_sample_target_batch.__get__(engine)
    events = []
    monkeypatch.setattr(dist, "all_reduce", lambda value, **kwargs: events.append(("vote", value.item())))
    engine._token_commit_callback = lambda chunk: events.append(("chunk", chunk["token_ids"]))
    engine._stream_delivered_counts = {}
    engine._stream_started = 0.0
    result = harness.run()
    assert events == [("vote", 0), ("chunk", [7])]
    assert result[0]["completion_token_ids"] == [7]


@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("remote", [False, True])
def test_backend_identity_failure_votes_before_publish_for_eager_and_graph(monkeypatch, eager, remote):
    engine = _engine()
    engine.config.enforce_eager = eager
    engine.config.spec_rhythm_roofline = ProfiledRoofline(
        {"1:1": 4},
        {
            "verification_attention_backends": ["fused_infer_attention_tree_v1"],
            "ar_attention_backend": "paged_attention_v1",
        },
        (),
    )
    output = {
        "used_aclgraph": True,
        "attention_backend": "fused_infer_attention_tree_v1" if remote else "dense_sdpa_tree_v1",
    }
    votes = []

    def vote(failed, **kwargs):
        votes.append(failed.item())
        failed.fill_(2)

    monkeypatch.setattr(dist, "all_reduce", vote)
    with pytest.raises(RuntimeError, match="attention backend validation failed") as error:
        engine._validate_spec_rhythm_target_graph(output, has_target_work=True)
    assert votes == [0 if remote else 2]
    assert (error.value.__cause__ is None) == remote


def _prefill_vote_worker(rank, world_size, bad_rank, store_path, queue):
    try:
        dist.init_process_group(
            "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world_size, timeout=timedelta(seconds=15)
        )
        engine = _engine(rank=rank)
        state = PearlPipelineState([1, 2], prompt_length=2, max_tokens=1, ignore_eos=True)
        engine._run_packed_hidden = lambda *args, **kwargs: torch.ones(1, 16)
        engine._run_packed_sample = lambda *args, **kwargs: torch.tensor([7])
        if rank == bad_rank:
            engine.model.layers[0].self_attn.tree_cache_nonfinite.fill_(True)
        error = None
        try:
            tokens = engine._prefill_and_sample_target_batch([[1, 2]], [state])
            state.token_ids.extend(tokens)
            state.committed_length += len(tokens)
        except RuntimeError as caught:
            error = str(caught)
        queue.put({"rank": rank, "error": error, "committed": state.committed_completion_token_ids})
    except Exception as error:
        queue.put({"rank": rank, "fatal": repr(error)})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires CPU Gloo")
@pytest.mark.parametrize("bad_rank", [0, 2])
def test_real_four_rank_prefill_vote_rejects_draft_or_target_fault(tmp_path, bad_rank):
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    processes = [
        context.Process(target=_prefill_vote_worker, args=(rank, 4, bad_rank, str(tmp_path / "store"), queue))
        for rank in range(4)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=20)
        assert all(not process.is_alive() and process.exitcode == 0 for process in processes)
        records = [queue.get(timeout=2) for _ in processes]
        assert {record["rank"] for record in records} == set(range(4))
        for record in records:
            assert "fatal" not in record, record
            assert "no first token was committed" in record["error"]
            assert not record["committed"]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        queue.close()
