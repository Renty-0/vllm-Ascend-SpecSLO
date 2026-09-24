#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from collections.abc import Callable
from typing import Any

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

env_variables: dict[str, Callable[[], Any]] = {
    # Optional local output directory for PEARL/SpecSLO NPU traces. Default:
    # None (disabled). Non-sensitive; opt-in diagnostic overhead only.
    "VLLM_ASCEND_PEARL_NPU_PROFILE_DIR": lambda: os.getenv("VLLM_ASCEND_PEARL_NPU_PROFILE_DIR"),
    # Trace rank: -1 selects all workers; otherwise a non-negative world rank.
    # Default None lets the execution path choose. Non-sensitive.
    "VLLM_ASCEND_PEARL_NPU_PROFILE_RANK": lambda: (
        int(os.environ["VLLM_ASCEND_PEARL_NPU_PROFILE_RANK"])
        if "VLLM_ASCEND_PEARL_NPU_PROFILE_RANK" in os.environ
        else None
    ),
    # Skip this many worker decode cycles before the profiler's own warmup
    # and active windows.  Online service traces need this to exclude initial
    # pipeline-fill cycles and capture steady-state draft/target concurrency.
    "VLLM_ASCEND_PEARL_NPU_PROFILE_WAIT_STEPS": lambda: int(
        os.getenv("VLLM_ASCEND_PEARL_NPU_PROFILE_WAIT_STEPS", "0")
    ),
    # PEARL/SpecSLO experimental diagnostics and compatibility switches.
    # All are non-sensitive. Production defaults keep diagnostics disabled
    # and select the native packed-tree/ACLGraph route.
    "VLLM_ASCEND_PEARL_ENABLE_TP3_MM_ALL_REDUCE": lambda: bool(
        int(os.getenv("VLLM_ASCEND_PEARL_ENABLE_TP3_MM_ALL_REDUCE", "0"))
    ),
    # Optional absolute directory for one-shot real-layer MC2 qualification
    # inputs. Empty (default) disables the probe. This non-secret diagnostic
    # must be used only with enforce-eager and removed before measurements.
    "VLLM_ASCEND_PEARL_MC2_CAPTURE_DIR": lambda: os.getenv(
        "VLLM_ASCEND_PEARL_MC2_CAPTURE_DIR", ""
    ),
    # Comma-separated positive, exact flattened M rows to capture. Empty
    # (default) disables the probe. Both capture variables must be set
    # together; ACLGraph capture and performance measurements are unsupported.
    "VLLM_ASCEND_PEARL_MC2_CAPTURE_ROWS": lambda: os.getenv(
        "VLLM_ASCEND_PEARL_MC2_CAPTURE_ROWS", ""
    ),
    # Projection family captured by the real-input probe. ``attention`` keeps
    # the historical o_proj behavior; ``down`` captures the MLP down_proj
    # inputs needed to qualify a cross-layer MC2 candidate. This selector is
    # diagnostic-only and has no effect while capture DIR/ROWS are unset.
    "VLLM_ASCEND_PEARL_MC2_CAPTURE_KIND": lambda: os.getenv(
        "VLLM_ASCEND_PEARL_MC2_CAPTURE_KIND", "attention"
    ),
    # Optional comma-separated decoder layer indices. Empty preserves the
    # historical one-shot behavior by capturing only the first matching
    # layer. Explicit indices may be used for cross-layer numerical audits.
    "VLLM_ASCEND_PEARL_MC2_CAPTURE_LAYERS": lambda: os.getenv(
        "VLLM_ASCEND_PEARL_MC2_CAPTURE_LAYERS", ""
    ),
    # Diagnostic-only escape hatch for a native-epilogue MC2 correctness A/B.
    # Default 0 keeps production fail-closed.  When set to 1, an exact
    # identity-bound ``tp3_matmul_allreduce+native_add_rmsnorm`` profile may
    # bypass only its repeated-p95 speed gate; hardware, TP size, exact shape,
    # source hash and numerical qualification remain mandatory.  Never use
    # this switch for throughput claims or production serving.
    "VLLM_ASCEND_PEARL_MC2_DIAGNOSTIC_FORCE_NATIVE_EPILOGUE": lambda: bool(
        int(os.getenv("VLLM_ASCEND_PEARL_MC2_DIAGNOSTIC_FORCE_NATIVE_EPILOGUE", "0"))
    ),
    # Experimental cross-layer target fusion.  This is deliberately default
    # off and is further fail-closed by the native model: only target TP3,
    # MC2-enabled, bias-free, uniform K=8576 FFNs with an exact qualified
    # profile entry may replace down_proj + all-reduce + the following RMSNorm.
    "VLLM_ASCEND_PEARL_MC2_DOWN_PROJ": lambda: bool(
        int(os.getenv("VLLM_ASCEND_PEARL_MC2_DOWN_PROJ", "0"))
    ),
    # Optional native-draft-only weight layout. -1 inherits the global
    # VLLM_ASCEND_ENABLE_NZ mode; 0/1 keep ND, 2 converts all supported
    # linears, 3/4/5 select both/down-only/gate-only FFN projections, and
    # 6/7 select down_proj on even/odd decoder layers.
    # This lets draft and target layouts be qualified independently.
    # 9 is an inference-only LM-head NZ policy; tied embeddings stay ND.
    "VLLM_ASCEND_PEARL_DRAFT_NZ_MODE": lambda: int(
        os.getenv("VLLM_ASCEND_PEARL_DRAFT_NZ_MODE", "-1")
    ),
    # Optional target-local layout override. Mode 3 converts both FFN
    # projections; mode 4 converts only down_proj, mode 5 only gate_up_proj,
    # modes 6/7 select down_proj on even/odd decoder layers, and mode 8 uses
    # NZ for QKV/O/down/LM-head while deliberately retaining gate-up in ND.
    # Mode 11 retains an ND down projection for verify graphs and adds an NZ
    # copy used only by larger mixed-prefill matrices.
    "VLLM_ASCEND_PEARL_TARGET_NZ_MODE": lambda: int(
        os.getenv("VLLM_ASCEND_PEARL_TARGET_NZ_MODE", "-1")
    ),
    "VLLM_ASCEND_PEARL_TARGET_LARGE_M_NZ_MIN_ROWS": lambda: int(
        os.getenv("VLLM_ASCEND_PEARL_TARGET_LARGE_M_NZ_MIN_ROWS", "192")
    ),
    # Strict qualification gate for native PARD eager execution.  Default off
    # keeps the public engine fail-closed.  This never enables draft ACLGraph;
    # it only permits the fixed-gamma greedy full-window NPU qualification run.
    "VLLM_ASCEND_SPECSLO_ENABLE_EXPERIMENTAL_PARD_EAGER": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECSLO_ENABLE_EXPERIMENTAL_PARD_EAGER", "0"))
    ),
    "VLLM_ASCEND_PEARL_VERBOSE": lambda: bool(int(os.getenv("VLLM_ASCEND_PEARL_VERBOSE", "0"))),
    "VLLM_ASCEND_PEARL_SYNC_GRAPH_INPUTS": lambda: bool(int(os.getenv("VLLM_ASCEND_PEARL_SYNC_GRAPH_INPUTS", "0"))),
    "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE": lambda: bool(
        int(os.getenv("VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE", "0"))
    ),
    # Opt in to production-style draft ACLGraph ordering: record graph-input
    # readiness, submit replay, then update every captured attention task on
    # the auxiliary stream. Valid values: 0 (default/off) or 1 (on).
    # Non-sensitive; experimental NPU performance switch.
    "VLLM_ASCEND_PEARL_DRAFT_REPLAY_FIRST_TASK_UPDATE": lambda: bool(
        int(os.getenv("VLLM_ASCEND_PEARL_DRAFT_REPLAY_FIRST_TASK_UPDATE", "0"))
    ),
    # Opt in to replay-first attention-task refreshes for the generic
    # ACLGraph runner used by the packed-FIA target path.  The graph is
    # submitted after an input-readiness dependency is queued, then every
    # captured attention task is refreshed through its ExternalEvent.  Valid
    # values: 0 (default/off) or 1 (on).  Non-sensitive; experimental NPU
    # performance switch.  This does not affect the multi-step tree-target
    # runner, whose execution contract is separate.
    "VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE": lambda: bool(
        int(os.getenv("VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE", "0"))
    ),
    # Coalesce only the ExternalEvent gate used by consecutive FIA task
    # updates in an exact target-verification ACLGraph.  A value of 1 keeps the
    # production one-event-per-layer contract (default/off); experimental
    # values 2 and 4 release that many independently updated single-operator
    # handles with one event.  CANN currently permits only one operator per
    # task-group handle, so this switch deliberately does not coalesce handles
    # or graph_task_update_begin/end calls.  Non-sensitive NPU performance
    # switch; changed-input numerical qualification is required before use.
    "VLLM_ASCEND_PEARL_TARGET_FIA_TASK_EVENT_GROUP_SIZE": lambda: int(
        os.getenv(
            "VLLM_ASCEND_PEARL_TARGET_FIA_TASK_EVENT_GROUP_SIZE",
            "1",
        )
    ),
    # Optionally release a shorter first FIA layer group before continuing
    # with the uniform group size above.  ``2 + 4 + ...`` preserves event4's
    # low steady-state host overhead while exposing the first target layers as
    # early as event2.  Zero disables the prefix and is the production
    # default.  Only exact target-verification graphs consume this setting.
    "VLLM_ASCEND_PEARL_TARGET_FIA_TASK_PREFIX_EVENT_GROUP_SIZE": lambda: int(
        os.getenv(
            "VLLM_ASCEND_PEARL_TARGET_FIA_TASK_PREFIX_EVENT_GROUP_SIZE",
            "0",
        )
    ),
    # Parallelize the host-side refresh of independent per-layer causal FIA
    # handles on target workers.  The default remains serial.  With grouped
    # events, every complete consecutive event group is assigned to exactly
    # one update stream and is released only after its final handle refresh.
    # Values above one remain an opt-in profiling primitive: target-stage
    # improvement alone did not improve the current end-to-end workload.
    "VLLM_ASCEND_PEARL_TARGET_FIA_TASK_UPDATE_WORKERS": lambda: int(
        os.getenv("VLLM_ASCEND_PEARL_TARGET_FIA_TASK_UPDATE_WORKERS", "1")
    ),
    "VLLM_ASCEND_PEARL_SYNC_GRAPH_TASK_UPDATE": lambda: bool(
        int(os.getenv("VLLM_ASCEND_PEARL_SYNC_GRAPH_TASK_UPDATE", "0"))
    ),
    "VLLM_ASCEND_PEARL_SYNC_GRAPH_REPLAY": lambda: bool(int(os.getenv("VLLM_ASCEND_PEARL_SYNC_GRAPH_REPLAY", "0"))),
    "VLLM_ASCEND_PEARL_VALIDATE_GRAPH_REPLAYS": lambda: bool(
        int(os.getenv("VLLM_ASCEND_PEARL_VALIDATE_GRAPH_REPLAYS", "0"))
    ),
    # Let every shape captured by one native PEARL graph runner use the same
    # allocator pool.  This is safe because a worker submits at most one model
    # graph at a time, and prevents each mixed-prefill bucket from retaining a
    # separate copy of all transformer intermediates.  Keep it opt-in until
    # the changed-shape replay and P128 performance gates have passed.
    "VLLM_ASCEND_PEARL_SHARED_GRAPH_POOL": lambda: bool(int(os.getenv("VLLM_ASCEND_PEARL_SHARED_GRAPH_POOL", "0"))),
    # Collect fine-grained host timings for native PagedAttention and fused
    # infer-attention graph-task refreshes. Disabled by default because the
    # per-task clock reads are intended for profiling, not production
    # throughput measurements.
    "VLLM_ASCEND_PEARL_PROFILE_PA_TASK_UPDATE": lambda: bool(
        int(os.getenv("VLLM_ASCEND_PEARL_PROFILE_PA_TASK_UPDATE", "0"))
    ),
    # Replace the TP1 draft worker's host-length CANN PagedAttention with a
    # device-position Triton kernel.  The captured graph then reads changing
    # sequence positions directly and owns no refreshable PA task handles.
    # This remains opt-in until complete-model numerical and service
    # throughput qualification pass.
    "VLLM_ASCEND_SPECSLO_DRAFT_DEVICE_PAGED_ATTENTION": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECSLO_DRAFT_DEVICE_PAGED_ATTENTION", "0"))
    ),
    # Reuse one host length signature and one workspace query per serial
    # draft step while preserving every per-layer graph task update.
    "VLLM_ASCEND_PEARL_DRAFT_STEP_MAJOR_PA_TASK_UPDATE": lambda: bool(
        int(
            os.getenv(
                "VLLM_ASCEND_PEARL_DRAFT_STEP_MAJOR_PA_TASK_UPDATE",
                "0",
            )
        )
    ),
    # Replace the fixed-gamma serial draft's one-token PA calls with a
    # request-major FULL-mask FIA contract.  Every request keeps its exact
    # visible prefix in the mask while the host-side KV length is rounded to
    # a power-of-two bucket.  This makes the captured FIA task arguments
    # stable inside a bucket and permits replay without rebuilding every
    # decoder layer's graph task.  Experimental and disabled by default until
    # the NPU numerical/throughput gates pass.
    "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BUCKET": lambda: bool(
        int(
            os.getenv(
                "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BUCKET",
                "0",
            )
        )
    ),
    # Learn fixed-gamma draft latency per physical ACLGraph bucket instead of
    # treating B1..B64 as one linear ms/token curve.  The graph family has a
    # pronounced launch/shape staircase, so the scalar model can otherwise
    # admit eager rows into a larger bucket that no longer fits target W.
    # Opt-in until the formal online workload A/B is complete.
    "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_BUCKET_TIMING": lambda: bool(
        int(
            os.getenv(
                "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_BUCKET_TIMING",
                "0",
            )
        )
    ),
    # Keep the host-side FIA KV-length tuple stable when different requests
    # occupy the same serial-draft graph bucket/lane.  All real rows use the
    # workload-wide lifetime envelope while the FULL mask continues to expose
    # only each row's exact causal prefix.  This is meaningful only together
    # with LINEAR_DRAFT_FIA_BUCKET and is independently opt-in while its NPU
    # numerical/performance gate is evaluated.
    "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_COMMON_KV": lambda: bool(
        int(
            os.getenv(
                "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_COMMON_KV",
                "0",
            )
        )
    ),
    # Replace the fixed-gamma common-KV draft graph's per-layer ExternalEvent
    # gates with one entry-wide stable-task barrier.  This is legal only when
    # every replay keeps the captured FIA query/KV length literals unchanged;
    # the native runner rejects a changed signature before graph submission.
    # Experimental and disabled by default pending the NPU ABI/numerical gate.
    "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_STABLE_TASK_BARRIER": lambda: bool(
        int(
            os.getenv(
                "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_STABLE_TASK_BARRIER",
                "0",
            )
        )
    ),
    # Use a fixed descending per-row lifetime-capacity vector for each serial
    # draft graph bucket. Active rows are ranked into dominating slots and
    # outputs are restored to caller order. This avoids making every row pay
    # the service-wide maximum KV scan while preserving a membership-invariant
    # graph signature. Independently opt-in and mutually exclusive with the
    # common-KV mode above.
    "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_RANKED_KV": lambda: bool(
        int(
            os.getenv(
                "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_RANKED_KV",
                "0",
            )
        )
    ),
    # Build every fixed-gamma FULL visibility mask in one broadcasted pass
    # while caching only the immutable arange base. Meaningful only with the
    # bucketed linear-draft FIA path; independently opt-in for NPU gating.
    "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BATCHED_MASKS": lambda: bool(
        int(
            os.getenv(
                "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BATCHED_MASKS",
                "0",
            )
        )
    ),
    # Reuse fixed-shape serial-draft positions, slot mappings, request tables
    # and FULL-mask storage across common-KV graph replays. Dynamic values are
    # rewritten on the current stream before the graph runner copies them into
    # graph-owned buffers. Experimental and disabled by default until the NPU
    # operator/stream-ordering gate passes.
    "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_PERSISTENT_STAGING": lambda: bool(
        int(
            os.getenv(
                "VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_PERSISTENT_STAGING",
                "0",
            )
        )
    ),
    # Priority of the auxiliary ACLGraph task-update stream. Ascend accepts
    # 0 for the default priority and -1 for the high-priority stream used by
    # TorchAir's production graph updater. Keep 0 as the compatibility
    # default until the native PEARL NPU A/B has passed.
    "VLLM_ASCEND_PEARL_GRAPH_UPDATE_STREAM_PRIORITY": lambda: int(
        os.getenv("VLLM_ASCEND_PEARL_GRAPH_UPDATE_STREAM_PRIORITY", "0")
    ),
    # Role-local overrides keep draft and target graph-update experiments
    # independently attributable. When unset they inherit the legacy global
    # switch above, preserving the established launcher contract.
    "VLLM_ASCEND_PEARL_DRAFT_GRAPH_UPDATE_STREAM_PRIORITY": lambda: int(
        os.getenv(
            "VLLM_ASCEND_PEARL_DRAFT_GRAPH_UPDATE_STREAM_PRIORITY",
            os.getenv("VLLM_ASCEND_PEARL_GRAPH_UPDATE_STREAM_PRIORITY", "0"),
        )
    ),
    "VLLM_ASCEND_PEARL_TARGET_GRAPH_UPDATE_STREAM_PRIORITY": lambda: int(
        os.getenv(
            "VLLM_ASCEND_PEARL_TARGET_GRAPH_UPDATE_STREAM_PRIORITY",
            os.getenv("VLLM_ASCEND_PEARL_GRAPH_UPDATE_STREAM_PRIORITY", "0"),
        )
    ),
    # Comma-separated request indices for bounded scheduler tracing.
    "VLLM_ASCEND_SPECRHYTHM_TRACE_REQUEST": lambda: os.getenv("VLLM_ASCEND_SPECRHYTHM_TRACE_REQUEST", ""),
    # Experimental Goodput-oriented earliest-deadline policy. Requests whose
    # final TPOT budget is still attainable are ordered by remaining terminal
    # slack; requests that have already exhausted that immutable budget no
    # longer displace an attainable request merely because their a_need grows.
    "VLLM_ASCEND_SPECRHYTHM_GOODPUT_EDF": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_GOODPUT_EDF", "0"))
    ),
    # Experimental target-side class priority.  When a fixed verification
    # envelope cannot consume every ready proposal, select the smallest TPOT
    # SLO first and retain a_need/urgency/age as tie breakers within a class.
    # This is deliberately independent of admission order: it changes only
    # which already-active request receives the next target opportunity.
    "VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST", "0"))
    ),
    # Restrict optional idle-window rolling continuations to requests in the
    # strictest TPOT class. Mandatory normal proposals and urgent a_need work
    # are unchanged; this only prevents relaxed rows from consuming residual
    # draft compute that could keep a tight row continuously ready.
    "VLLM_ASCEND_SPECRHYTHM_RESIDUAL_EAGER_TIGHT_ONLY": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_RESIDUAL_EAGER_TIGHT_ONLY", "0"))
    ),
    # Admit arrival-ready requests with the smallest TPOT SLO first.  The
    # paper's TPOT clock starts after prefill/first-token admission, so a
    # relaxed request can safely remain pending instead of occupying a scarce
    # active decode row ahead of strict traffic.
    "VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION", "0"))
    ),
    # Limit the admission-wait term used by a_need to this many request-local
    # TPOT intervals. Zero preserves the established unbounded-debt policy.
    # This does not alter measured TPOT, only the online scheduling signal.
    "VLLM_ASCEND_SPECRHYTHM_ARRIVAL_DEBT_CAP_TOKENS": lambda: int(
        os.getenv("VLLM_ASCEND_SPECRHYTHM_ARRIVAL_DEBT_CAP_TOKENS", "0")
    ),
    # Soft active-set caps for relaxed traffic.  A cap is enforced only while
    # a tighter-class request remains pending, and is released for tail drain.
    # Zero disables the corresponding cap.
    "VLLM_ASCEND_SPECRHYTHM_NORMAL_ACTIVE_CAP": lambda: int(
        os.getenv("VLLM_ASCEND_SPECRHYTHM_NORMAL_ACTIVE_CAP", "0")
    ),
    "VLLM_ASCEND_SPECRHYTHM_LOOSE_ACTIVE_CAP": lambda: int(
        os.getenv("VLLM_ASCEND_SPECRHYTHM_LOOSE_ACTIVE_CAP", "0")
    ),
    # Allow relaxed requests to borrow otherwise idle active slots, then
    # suspend them at a cycle boundary when a coalesced tight cohort arrives.
    # Their KV and request state remain resident and their TPOT clock keeps
    # running; they resume without another prefill when a slot is available.
    "VLLM_ASCEND_SPECRHYTHM_PREEMPT_LOOSE_FOR_TIGHT": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_PREEMPT_LOOSE_FOR_TIGHT", "0"))
    ),
    # Number of consecutive urgent-tight target cycles that may defer ready
    # loose proposals before one mandatory release cycle. Gamma remains fixed;
    # this only reshapes the per-cycle verification cohort.
    "VLLM_ASCEND_SPECRHYTHM_TIGHT_TARGET_BURST": lambda: int(
        os.getenv("VLLM_ASCEND_SPECRHYTHM_TIGHT_TARGET_BURST", "0")
    ),
    # Optional asymmetric logical-home capacity for SLO-partitioned dual
    # batches. 0.5 preserves the established balanced 32/32 split at B64.
    "VLLM_ASCEND_SPECRHYTHM_TIGHT_HOME_FRACTION": lambda: float(
        os.getenv("VLLM_ASCEND_SPECRHYTHM_TIGHT_HOME_FRACTION", "0.5")
    ),
    "VLLM_ASCEND_SPECRHYTHM_TREE_GRAPH": lambda: bool(int(os.getenv("VLLM_ASCEND_SPECRHYTHM_TREE_GRAPH", "1"))),
    "VLLM_ASCEND_SPECRHYTHM_USE_FIA": lambda: bool(int(os.getenv("VLLM_ASCEND_SPECRHYTHM_USE_FIA", "0"))),
    # Fuse a staged whole-prompt target prefill with the current fixed-gamma
    # verification into one eager packed-FIA model pass. This deliberately
    # retains the existing single-stream/HCCL ordering and remains opt-in
    # until its NPU numerical and performance gates have passed.
    "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_PREFILL": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_PREFILL", "0"))
    ),
    # Experimental fixed-envelope ACLGraph for the mixed verification/prefill
    # target pass. Keep this separate from MIXED_TARGET_PREFILL so enabling
    # the eager fusion cannot silently change capture or replay behavior.
    "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH", "0"))
    ),
    # Keep the mixed verification+prefill graph family independent from the
    # ordinary decode-only target graph family.  The latter already has a
    # numerically-qualified causal chain; capturing another exact FIA graph
    # for every request count wastes 32 entries and a large amount of HBM.
    "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_PREFILL_ONLY": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_PREFILL_ONLY", "0"))
    ),
    # Zero preserves the ordinary graph runner capacity. A mixed-target graph
    # caller must explicitly provision enough packed-token capacity (for
    # example 645 for gamma=4 and the 512-token prompt bucket).
    "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_MAX_TOKENS": lambda: int(
        os.getenv("VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_MAX_TOKENS", "0")
    ),
    # Optional comma-separated subset of the production prompt-token buckets.
    # Keeping only workload-relevant buckets bounds resident ACLGraph HBM;
    # the scheduler caps each staged mixed prefill to the largest selected
    # bucket so omitted shapes never become an eager fallback.
    "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_BUCKETS": lambda: os.getenv(
        "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_BUCKETS", ""
    ),
    # Optional comma-separated subset of the supported verification-row
    # capacities (16,24,32,48,64).  High-load services can keep only one
    # upper capacity
    # and spend the saved graph entries on finer prompt-token buckets, which
    # avoids extreme prompt padding without changing the fixed graph shape.
    "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_VERIFY_CAPACITIES": lambda: os.getenv(
        "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_VERIFY_CAPACITIES",
        "",
    ),
    # Number of newly arrived prompt rows retained by one stable mixed
    # verification+prefill graph. Four preserves the historical coalesce4
    # envelope; larger explicitly provisioned cohorts amortize token-chunk
    # boundary waste without changing any real prompt length.
    "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_PROMPT_CAPACITY": lambda: int(
        os.getenv(
            "VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_PROMPT_CAPACITY",
            "4",
        )
    ),
    # Permit one bounded staged prompt cohort to own KV pages before a decode
    # slot becomes vacant.  Its first token remains private until an
    # incumbent completes, so service batch capacity and TPOT semantics are
    # unchanged while prompt work can be hidden behind full-batch decode.
    "VLLM_ASCEND_SPECRHYTHM_PREFETCH_AHEAD": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_PREFETCH_AHEAD", "0"))
    ),
    # Highest exact-Q ordinary target-verify graph captured ahead of sealed
    # service.  Keep the historical 32-row family by default; services that
    # deliberately admit larger target batches must raise this together with
    # their graph-entry budget so arrival jitter cannot discover a new shape.
    "VLLM_ASCEND_SPECRHYTHM_STABLE_TARGET_VERIFY_MAX_CAPACITY": lambda: int(
        os.getenv(
            "VLLM_ASCEND_SPECRHYTHM_STABLE_TARGET_VERIFY_MAX_CAPACITY",
            "32",
        )
    ),
    # Optional request-row bucket for ordinary fixed-gamma target graphs.
    # gamma=4 and bucket=4 make every target GEMM row count a multiple of 16,
    # which selects the faster 910B2 down-projection tiling.  The default
    # remains exact-Q until the padded family passes full service A/B.
    "VLLM_ASCEND_SPECRHYTHM_STABLE_TARGET_VERIFY_REQUEST_BUCKET": lambda: int(
        os.getenv(
            "VLLM_ASCEND_SPECRHYTHM_STABLE_TARGET_VERIFY_REQUEST_BUCKET",
            "1",
        )
    ),
    # Experimental target-only GEMM row padding.  Each native linear pads its
    # packed token matrix to this multiple and slices the result immediately,
    # keeping attention/KV semantics unchanged while making exact and bucketed
    # verification graphs use the same matrix shapes.
    "VLLM_ASCEND_PEARL_TARGET_TOKEN_PAD_MULTIPLE": lambda: int(
        os.getenv("VLLM_ASCEND_PEARL_TARGET_TOKEN_PAD_MULTIPLE", "1")
    ),
    # Experimental TP3 load balancing.  A positive, tile-aligned value moves
    # this many FFN channels from each of target ranks 0/1 to rank 2, while
    # attention uses the exact 3/3/2 KV-group partition instead of 3/3/3
    # zero padding.  Zero preserves the production partition.
    "VLLM_ASCEND_PEARL_TARGET_TP3_BALANCED_FFN_SHIFT": lambda: int(
        os.getenv("VLLM_ASCEND_PEARL_TARGET_TP3_BALANCED_FFN_SHIFT", "0")
    ),
    "VLLM_ASCEND_PEARL_TARGET_TP3_LIGHT_RANK": lambda: int(
        os.getenv("VLLM_ASCEND_PEARL_TARGET_TP3_LIGHT_RANK", "-1")
    ),
    # Pair the target fusion above with a draft-side mixed first step: staged
    # prompt rows share the first eager FIA model pass with incumbent draft
    # roots, then a qualified three-step PA graph completes gamma=4.
    "VLLM_ASCEND_SPECRHYTHM_MIXED_DRAFT_PREFILL": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_MIXED_DRAFT_PREFILL", "0"))
    ),
    # Preserve the incumbent PA full-chain exactly, but submit a disjoint
    # staged draft prompt prefill on a side NPU stream while that graph runs.
    # The service joins the stream before its existing cross-model broadcast
    # fence, so no partially populated prompt KV row can be published.
    "VLLM_ASCEND_SPECRHYTHM_OVERLAP_DRAFT_PREFILL": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_OVERLAP_DRAFT_PREFILL", "0"))
    ),
    # Greedy target ranks compute an identical compact verdict. Materialize
    # it once on the target leader and fan it out through the existing Gloo
    # coordination group instead of launching a second tiny HCCL collective.
    "VLLM_ASCEND_SPECRHYTHM_GLOO_CORRECTION": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_GLOO_CORRECTION", "0"))
    ),
    # Publish the two per-cycle float64 accounting/frontier envelopes through
    # the existing CPU coordination group. This is valid only for TP1 draft,
    # where that group contains every world rank; other topologies retain the
    # WORLD/HCCL path. Experimental and disabled by default.
    "VLLM_ASCEND_SPECRHYTHM_GLOO_ACCOUNTING": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_GLOO_ACCOUNTING", "0"))
    ),
    # Move the compact fixed-gamma draft proposal through the existing CPU
    # coordination group.  This lets target TP compute remain queued while
    # the proposal is exchanged, instead of synchronizing every target
    # stream before an overlapping-group HCCL broadcast.  The corresponding
    # correction must also use Gloo so no cross-model HCCL collective is
    # posted while target TP all-reduces are still in flight.
    "VLLM_ASCEND_SPECRHYTHM_GLOO_PROPOSAL": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_GLOO_PROPOSAL", "0"))
    ),
    # Reuse the fixed-shape device and pinned-host source buffers used by the
    # compact Gloo proposal path. The synchronous Gloo broadcast releases
    # both buffers before the next cycle.
    "VLLM_ASCEND_SPECRHYTHM_GLOO_PROPOSAL_REUSE_SOURCE_BUFFERS": lambda: bool(
        int(
            os.getenv(
                "VLLM_ASCEND_SPECRHYTHM_GLOO_PROPOSAL_REUSE_SOURCE_BUFFERS",
                "0",
            )
        )
    ),
    # Fixed-gamma linear drafting already observes accepted/proposed tokens
    # after every verification round.  Let that measured acceptance EMA be
    # the sole rolling-eager benefit estimate, avoiding four full-vocabulary
    # softmax reductions in each gamma-4 draft graph.  Tree budget shaping
    # retains exact per-node confidence and is intentionally unaffected.
    "VLLM_ASCEND_SPECRHYTHM_LINEAR_ACCEPTANCE_ONLY": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_LINEAR_ACCEPTANCE_ONLY", "0"))
    ),
    "VLLM_ASCEND_SPECRHYTHM_VALIDATE_MAILBOX": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_VALIDATE_MAILBOX", "0"))
    ),
    "VLLM_ASCEND_SPECRHYTHM_FORCE_STEPWISE_TARGET": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_FORCE_STEPWISE_TARGET", "0"))
    ),
    "VLLM_ASCEND_SPECRHYTHM_STEPWISE_TARGET_FIA": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_STEPWISE_TARGET_FIA", "0"))
    ),
    "VLLM_ASCEND_SPECRHYTHM_PACKED_TARGET": lambda: bool(int(os.getenv("VLLM_ASCEND_SPECRHYTHM_PACKED_TARGET", "0"))),
    # Run one same-shape eager FIA causality probe per target worker.  The
    # probe perturbs only the last queried proposal token, checks that earlier
    # rows are unchanged, and restores the original KV row before returning.
    # Diagnostic only; disabled by default because it executes three forwards.
    "VLLM_ASCEND_SPECRHYTHM_VALIDATE_PACKED_CAUSAL_LEAKAGE": lambda: bool(
        int(
            os.getenv(
                "VLLM_ASCEND_SPECRHYTHM_VALIDATE_PACKED_CAUSAL_LEAKAGE",
                "0",
            )
        )
    ),
    "VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH", "0"))
    ),
    "VLLM_ASCEND_USE_NATIVE_QWEN2_ROPE": lambda: bool(int(os.getenv("VLLM_ASCEND_USE_NATIVE_QWEN2_ROPE", "0"))),
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to enable MatmulAllReduce fusion kernel when tensor parallel is enabled.
    # this feature is supported in A2, and eager mode will get better performance.
    "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", "0"))),
    # Whether to enable FlashComm optimization when tensor parallel is enabled.
    # This feature will get better performance when concurrency is large.
    # DEPRECATED: use additional_config.enable_flashcomm1 instead.
    "VLLM_ASCEND_ENABLE_FLASHCOMM1": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0"))),
    # Whether to enable FLASHCOMM2. Setting it to 0 disables the feature, while setting it to 1 or above enables it.
    # The specific value set will be used as the O-matrix TP group size for flashcomm2.
    # For a detailed introduction to the parameters and the differences and applicable scenarios
    # between this feature and FLASHCOMM1, please refer to the feature guide in the documentation.
    "VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": lambda: int(os.getenv("VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE", 0)),
    # Whether to enable msMonitor tool to monitor the performance of vllm-ascend.
    "MSMONITOR_USE_DAEMON": lambda: bool(int(os.getenv("MSMONITOR_USE_DAEMON", "0"))),
    # Whether to enable MLAPO optimization for DeepSeek W8A8 series models.
    # This option is enabled by default. MLAPO can improve performance, but
    # it will consume more NPU memory. If reducing NPU memory usage is a higher priority
    # for your DeepSeek W8A8 scene, then disable it.
    "VLLM_ASCEND_ENABLE_MLAPO": lambda: bool(int(os.getenv("VLLM_ASCEND_ENABLE_MLAPO", "1"))),
    # Whether to enable weight cast format to FRACTAL_NZ.
    # 0: close nz;
    # 1: only quant case enable nz;
    # 2: enable nz as long as possible.
    "VLLM_ASCEND_ENABLE_NZ": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_NZ", 1)),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Whether to enable fused MC2 (`dispatch_gmm_combine_decode` / `dispatch_ffn_combine`).
    # 0, or not set: default ALLTOALL and MC2 will be used.
    # 1: ALLTOALL and MC2 might be replaced by `dispatch_ffn_combine` operator.
    # `dispatch_ffn_combine` can be used only for moe layer with W8A8, EP<=32, non-mtp, non-dynamic-eplb.
    # 2: MC2 might be replaced by `dispatch_gmm_combine_decode` operator.
    # `dispatch_gmm_combine_decode` can be used only for **decode node** moe layer
    # with W8A8. And MTP layer must be W8A8.
    "VLLM_ASCEND_ENABLE_FUSED_MC2": lambda: int(os.getenv("VLLM_ASCEND_ENABLE_FUSED_MC2", "0")),
    # DEPRECATED: VLLM_ASCEND_BALANCE_SCHEDULING env var will be removed in a future release.
    # Use --additional-config '{"enable_balance_scheduling": true}' instead.
    "VLLM_ASCEND_BALANCE_SCHEDULING": lambda: bool(int(os.getenv("VLLM_ASCEND_BALANCE_SCHEDULING", "0"))),
    # Whether to enable utility-based victim selection in scheduler preemption.
    "VLLM_ASCEND_ENABLE_UTILITY_VICTIM_SELECTION": lambda: bool(
        int(os.getenv("VLLM_ASCEND_ENABLE_UTILITY_VICTIM_SELECTION", "0"))
    ),
    # Emergency kill switch for utility-based victim selection.
    "VLLM_ASCEND_UTILITY_KILL_SWITCH": lambda: bool(int(os.getenv("VLLM_ASCEND_UTILITY_KILL_SWITCH", "0"))),
    # Completion factor weight in utility delta calculation.
    "VLLM_ASCEND_UTILITY_COMPLETION_WEIGHT": lambda: float(os.getenv("VLLM_ASCEND_UTILITY_COMPLETION_WEIGHT", "0.5")),
    # Preemption-count factor weight in utility delta calculation.
    "VLLM_ASCEND_UTILITY_PREEMPT_WEIGHT": lambda: float(os.getenv("VLLM_ASCEND_UTILITY_PREEMPT_WEIGHT", "0.3")),
    # Minimum KV utilization ratio required to enable utility ranking.
    "VLLM_ASCEND_UTILITY_KV_GATE": lambda: float(os.getenv("VLLM_ASCEND_UTILITY_KV_GATE", "0.0")),
    # Cooldown window (seconds) between two utility-based victim selections.
    "VLLM_ASCEND_UTILITY_COOLDOWN_S": lambda: float(os.getenv("VLLM_ASCEND_UTILITY_COOLDOWN_S", "0.0")),
    # Minimum running queue size required before enabling utility-based victim selection.
    "VLLM_ASCEND_UTILITY_MIN_RUNNING": lambda: int(os.getenv("VLLM_ASCEND_UTILITY_MIN_RUNNING", "1")),
    # Whether to capture shared-snapshot counterfactual records for utility decisions.
    "VLLM_ASCEND_UTILITY_SNAPSHOT_ENABLED": lambda: bool(int(os.getenv("VLLM_ASCEND_UTILITY_SNAPSHOT_ENABLED", "0"))),
    # Number of top-ranked candidates to keep in each utility decision snapshot.
    "VLLM_ASCEND_UTILITY_SNAPSHOT_TOP_K": lambda: int(os.getenv("VLLM_ASCEND_UTILITY_SNAPSHOT_TOP_K", "3")),
    # Number of recent utility decision snapshots retained in memory.
    "VLLM_ASCEND_UTILITY_SNAPSHOT_HISTORY_SIZE": lambda: int(
        os.getenv("VLLM_ASCEND_UTILITY_SNAPSHOT_HISTORY_SIZE", "32")
    ),
    # use fused op transpose_kv_cache_by_block, default is True
    "VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK": lambda: bool(
        int(os.getenv("VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK", "1"))
    ),
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
    # Disable AddRmsNormBias custom-op dependent fusion passes. This is useful
    # when the installed libopapi.so does not export the required symbols.
    "VLLM_ASCEND_DISABLE_ADD_RMS_NORM_BIAS_CUSTOM_OP": lambda: bool(
        int(os.getenv("VLLM_ASCEND_DISABLE_ADD_RMS_NORM_BIAS_CUSTOM_OP", "0"))
    ),
    # Disable the AscendC top-k/top-p custom op and use the PyTorch fallback.
    "VLLM_ASCEND_DISABLE_TOP_K_TOP_P_CUSTOM_OP": lambda: bool(
        int(os.getenv("VLLM_ASCEND_DISABLE_TOP_K_TOP_P_CUSTOM_OP", "0"))
    ),
    # Optional JSONL destination for Python attention dispatch diagnostics.
    # Empty (default): disabled. Non-sensitive path. Capture events do not
    # count graph replays.
    "VLLM_ASCEND_ATTENTION_PATH_PROBE_JSONL": lambda: os.getenv("VLLM_ASCEND_ATTENTION_PATH_PROBE_JSONL", ""),
    # Maximum detailed dispatch records. Valid range: >= 0. Zero selects
    # summary-only mode.
    "VLLM_ASCEND_ATTENTION_PATH_PROBE_MAX_RECORDS": lambda: int(
        os.getenv("VLLM_ASCEND_ATTENTION_PATH_PROBE_MAX_RECORDS", "2048")
    ),
    # Maximum bytes accepted for detailed records. Valid range: >= 0. Zero
    # selects summary-only mode.
    "VLLM_ASCEND_ATTENTION_PATH_PROBE_MAX_BYTES": lambda: int(
        os.getenv("VLLM_ASCEND_ATTENTION_PATH_PROBE_MAX_BYTES", "1048576")
    ),
    # Record one detailed row every N Python attention dispatches. Valid
    # range: >= 1.
    "VLLM_ASCEND_ATTENTION_PATH_PROBE_EVERY": lambda: int(os.getenv("VLLM_ASCEND_ATTENTION_PATH_PROBE_EVERY", "64")),
    # Number of records buffered before a batched write. Valid range: >= 1.
    "VLLM_ASCEND_ATTENTION_PATH_PROBE_BUFFER_RECORDS": lambda: int(
        os.getenv("VLLM_ASCEND_ATTENTION_PATH_PROBE_BUFFER_RECORDS", "256")
    ),
    # A single process owns the output file; valid range is a global rank in
    # [0, WORLD_SIZE). Defaults to global rank zero.
    "VLLM_ASCEND_ATTENTION_PATH_PROBE_OWNER_RANK": lambda: int(
        os.getenv("VLLM_ASCEND_ATTENTION_PATH_PROBE_OWNER_RANK", "0")
    ),
    # Shared non-secret run identifier used to join core and platform
    # telemetry artifacts. Valid format: 1-64 letters, digits, '.', '_', '-';
    # the first character must be alphanumeric.
    "VLLM_TELEMETRY_RUN_ID": lambda: os.getenv("VLLM_TELEMETRY_RUN_ID", ""),
    # -- Sim-LLM: KV reuse optimization ---------------------------------------
    # Whether to enable Sim-LLM KV reuse optimization. When set to 1, the Sim-LLM
    # patch wraps NPUModelRunner.execute_model() at worker init time.
    # 0: disabled (default), 1: enabled.
    "VLLM_ASCEND_SIMLLM_ENABLED": lambda: bool(int(os.getenv("VLLM_ASCEND_SIMLLM_ENABLED", "0"))),
    # Cosine similarity threshold for KV reuse match. Embeddings with cosine
    # similarity >= this value are considered a match. Paper default 0.8.
    # Valid range: [0.0, 1.0]. Higher values = stricter matching, fewer KV reuses.
    "VLLM_ASCEND_SIMLLM_COSINE_THRESHOLD": lambda: float(os.getenv("VLLM_ASCEND_SIMLLM_COSINE_THRESHOLD", "0.8")),
    # Number of bits for SimHash LSH projection. More bits = fewer collisions
    # but larger hash storage. Paper default 64 (fits in a single int64).
    # Valid range: [16, 256]. Recommended: 32, 64, or 128.
    "VLLM_ASCEND_SIMLLM_LSH_NUM_BITS": lambda: int(os.getenv("VLLM_ASCEND_SIMLLM_LSH_NUM_BITS", "64")),
    # Batch size threshold for switching from exhaustive cosine to LSH bucket
    # merge strategy. Below this threshold: exact cosine per candidate.
    # At or above: LSH bucket membership with KV merging. Default 32.
    "VLLM_ASCEND_SIMLLM_LSH_BATCH_THRESHOLD": lambda: int(os.getenv("VLLM_ASCEND_SIMLLM_LSH_BATCH_THRESHOLD", "32")),
    # Maximum number of cached tasks in KV_Manager. When exceeded, the
    # least-recently-accessed task is evicted (O(1) via OrderedDict).
    # Default 1024. Increase for higher reuse rates on diverse workloads.
    "VLLM_ASCEND_SIMLLM_KV_CACHE_SIZE": lambda: int(os.getenv("VLLM_ASCEND_SIMLLM_KV_CACHE_SIZE", "1024")),
    # Number of bottom (early) transformer layers whose KV is retained in the
    # sandwich config for unmatched tasks. Default 3 (layers 0, 1, 2).
    "VLLM_ASCEND_SIMLLM_SANDWICH_BOTTOM": lambda: int(os.getenv("VLLM_ASCEND_SIMLLM_SANDWICH_BOTTOM", "3")),
    # Number of top (late) transformer layers whose KV is retained in the
    # sandwich config for unmatched tasks. Default 3 (layers L-3 .. L-1).
    # Total KV retention = (bottom + top) / num_layers.
    "VLLM_ASCEND_SIMLLM_SANDWICH_TOP": lambda: int(os.getenv("VLLM_ASCEND_SIMLLM_SANDWICH_TOP", "3")),
    # Pooling strategy for task embedding extraction from hidden states.
    # Options: "mean" (mean pooling over sequence), "last" (last token only),
    # "cls" (first token / CLS token). Default "mean".
    "VLLM_ASCEND_SIMLLM_EMBEDDING_POOLING": lambda: os.getenv("VLLM_ASCEND_SIMLLM_EMBEDDING_POOLING", "mean"),
    # Batch match ratio threshold for deferral logic. If the fraction of matched
    # tasks in a batch exceeds this value, unmatched tasks are deferred to the
    # next scheduling cycle. Valid range: [0.0, 1.0]. Default 0.5.
    "VLLM_ASCEND_SIMLLM_DEFERRAL_RATIO": lambda: float(os.getenv("VLLM_ASCEND_SIMLLM_DEFERRAL_RATIO", "0.5")),
    # Maximum number of times a task can be deferred before being force-processed
    # regardless of match status. Guards against starvation. Default 3.
    "VLLM_ASCEND_SIMLLM_MAX_DEFERRALS": lambda: int(os.getenv("VLLM_ASCEND_SIMLLM_MAX_DEFERRALS", "3")),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
