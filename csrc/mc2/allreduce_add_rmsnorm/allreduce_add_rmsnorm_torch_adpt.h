/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ALLREDUCE_ADD_RMSNORM_TORCH_ADPT_H
#define ALLREDUCE_ADD_RMSNORM_TORCH_ADPT_H

namespace vllm_ascend {

constexpr int64_t ALLREDUCE_ADD_RMSNORM_CHAIN_STATE_ROWS = 64;
constexpr int64_t ALLREDUCE_ADD_RMSNORM_CHAIN_STATE_WORDS = 4;

inline void validate_allreduce_add_rmsnorm_inputs(
    const at::Tensor &local_projection,
    const at::Tensor &residual,
    const at::Tensor &gamma,
    const at::Tensor &chain_state,
    double epsilon)
{
    TORCH_CHECK(
        epsilon == 1.0e-6,
        "experimental allreduce_add_rmsnorm is qualified only for epsilon=1e-6");
    TORCH_CHECK(
        local_projection.sizes() == residual.sizes(),
        "local_projection and residual must have identical shapes");
    TORCH_CHECK(
        chain_state.scalar_type() == at::kLong && chain_state.dim() == 2 &&
            chain_state.size(0) == ALLREDUCE_ADD_RMSNORM_CHAIN_STATE_ROWS &&
            chain_state.size(1) == ALLREDUCE_ADD_RMSNORM_CHAIN_STATE_WORDS,
        "allreduce_add_rmsnorm chain_state must be int64[64, 4]");
    TORCH_CHECK(
        chain_state.device() == local_projection.device() && chain_state.is_contiguous(),
        "allreduce_add_rmsnorm chain_state must be contiguous on the projection device");
}

// chain_state is an explicit device-side handoff between consecutive,
// fixed-shape epilogues on one serialized TP group.  Start a chain with
// zeros, feed each next_chain_state into the following call, and set
// flush_chain on the final call before an ACLGraph replay boundary.
std::tuple<at::Tensor, at::Tensor, at::Tensor> allreduce_add_rmsnorm_chained(
    const at::Tensor &local_projection,
    const at::Tensor &residual,
    const at::Tensor &gamma,
    const at::Tensor &chain_state,
    c10::string_view group_tp,
    int64_t tp_rank_size,
    int64_t tp_rank_id,
    double epsilon,
    bool is_gather_add_out,
    bool flush_chain)
{
    validate_allreduce_add_rmsnorm_inputs(
        local_projection, residual, gamma, chain_state, epsilon);

    at::Tensor output = at::empty_like(residual);
    at::Tensor add_out = at::empty_like(residual);
    at::Tensor next_chain_state = at::empty_like(chain_state);
    std::string group_tp_str(group_tp);
    char *group_tp_ptr = group_tp_str.data();
    EXEC_NPU_CMD(aclnnAllreduceAddRmsnorm,
        local_projection,
        residual,
        gamma,
        chain_state,
        group_tp_ptr,
        tp_rank_size,
        tp_rank_id,
        epsilon,
        is_gather_add_out,
        flush_chain,
        output,
        add_out,
        next_chain_state);
    return {output, add_out, next_chain_state};
}

std::tuple<at::Tensor, at::Tensor> allreduce_add_rmsnorm(
    const at::Tensor &local_projection,
    const at::Tensor &residual,
    const at::Tensor &gamma,
    c10::string_view group_tp,
    int64_t tp_rank_size,
    int64_t tp_rank_id,
    double epsilon,
    bool is_gather_add_out)
{
    // Preserve the original two-output API as one standalone, fully flushed
    // invocation. Chained production execution supplies persistent state to
    // allreduce_add_rmsnorm_chained instead.
    at::Tensor chain_state = at::zeros(
        {ALLREDUCE_ADD_RMSNORM_CHAIN_STATE_ROWS,
         ALLREDUCE_ADD_RMSNORM_CHAIN_STATE_WORDS},
        local_projection.options().dtype(at::kLong));
    auto chained = allreduce_add_rmsnorm_chained(
        local_projection,
        residual,
        gamma,
        chain_state,
        group_tp,
        tp_rank_size,
        tp_rank_id,
        epsilon,
        is_gather_add_out,
        true);
    return {std::get<0>(chained), std::get<1>(chained)};
}

}  // namespace vllm_ascend
#endif
