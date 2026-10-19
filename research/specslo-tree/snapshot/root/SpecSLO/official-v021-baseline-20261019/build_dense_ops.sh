#!/usr/bin/env bash
set -euo pipefail
root=/root/SpecSLO/official-v021-baseline-20261019
# Same CPATH include as the original official build_aclnn.sh A2 branch.
export CPATH="$root/vllm-ascend/csrc/third_party/catlass/include${CPATH:+:$CPATH}"
export MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 SOC_VERSION=ascend910b1
cd "$root/vllm-ascend/csrc"
exec bash "$root/env.sh" env CPATH="$CPATH" MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 SOC_VERSION=ascend910b1 bash build.sh --pkg \
 '--ops=scatter_nd_update_v2;matmul_allreduce_add_rmsnorm;add_rms_norm_bias;apply_top_k_top_p_custom;transpose_kv_cache_by_block;reshape_and_cache_bnsd;store_kv_block;inplace_partial_rotary_mul' \
 --soc=ascend910b
