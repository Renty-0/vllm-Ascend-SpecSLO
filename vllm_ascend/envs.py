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
        if "VLLM_ASCEND_PEARL_NPU_PROFILE_RANK" in os.environ else None
    ),
    # PEARL/SpecSLO experimental diagnostics and compatibility switches.
    # All are non-sensitive. Production defaults keep diagnostics disabled
    # and select the native packed-tree/ACLGraph route.
    "VLLM_ASCEND_PEARL_ENABLE_TP3_MM_ALL_REDUCE": lambda: bool(
        int(os.getenv("VLLM_ASCEND_PEARL_ENABLE_TP3_MM_ALL_REDUCE", "0"))
    ),
    "VLLM_ASCEND_PEARL_VERBOSE": lambda: bool(int(os.getenv("VLLM_ASCEND_PEARL_VERBOSE", "0"))),
    "VLLM_ASCEND_PEARL_SYNC_GRAPH_INPUTS": lambda: bool(
        int(os.getenv("VLLM_ASCEND_PEARL_SYNC_GRAPH_INPUTS", "0"))
    ),
    "VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE": lambda: bool(
        int(os.getenv("VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE", "0"))
    ),
    "VLLM_ASCEND_PEARL_SYNC_GRAPH_TASK_UPDATE": lambda: bool(
        int(os.getenv("VLLM_ASCEND_PEARL_SYNC_GRAPH_TASK_UPDATE", "0"))
    ),
    "VLLM_ASCEND_PEARL_SYNC_GRAPH_REPLAY": lambda: bool(
        int(os.getenv("VLLM_ASCEND_PEARL_SYNC_GRAPH_REPLAY", "0"))
    ),
    "VLLM_ASCEND_PEARL_VALIDATE_GRAPH_REPLAYS": lambda: bool(
        int(os.getenv("VLLM_ASCEND_PEARL_VALIDATE_GRAPH_REPLAYS", "0"))
    ),
    # Comma-separated request indices for bounded scheduler tracing.
    "VLLM_ASCEND_SPECRHYTHM_TRACE_REQUEST": lambda: os.getenv(
        "VLLM_ASCEND_SPECRHYTHM_TRACE_REQUEST", ""
    ),
    "VLLM_ASCEND_SPECRHYTHM_TREE_GRAPH": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_TREE_GRAPH", "1"))
    ),
    "VLLM_ASCEND_SPECRHYTHM_USE_FIA": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_USE_FIA", "0"))
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
    "VLLM_ASCEND_SPECRHYTHM_PACKED_TARGET": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_PACKED_TARGET", "0"))
    ),
    "VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH": lambda: bool(
        int(os.getenv("VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH", "0"))
    ),
    "VLLM_ASCEND_USE_NATIVE_QWEN2_ROPE": lambda: bool(
        int(os.getenv("VLLM_ASCEND_USE_NATIVE_QWEN2_ROPE", "0"))
    ),
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
