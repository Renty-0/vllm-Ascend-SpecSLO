#include <kernel_operator.h>
#include "allreduce_add_rmsnorm_aiv_kernel.h"

extern "C" __global__ __aicore__ void allreduce_add_rmsnorm(
    GM_ADDR local_projection,
    GM_ADDR residual,
    GM_ADDR gamma,
    GM_ADDR chain_state,
    GM_ADDR y,
    GM_ADDR add_out,
    GM_ADDR next_chain_state,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AllreduceAddRmsnormTilingData);
    GET_TILING_DATA(tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    Hccl<HCCL_SERVER_TYPE_AICPU> hccl;
    auto *tilingData = reinterpret_cast<__gm__ AllreduceAddRmsnormTilingData *>(tiling);
    __gm__ void *mc2InitTiling = reinterpret_cast<__gm__ void *>(&tilingData->mc2InitTiling);
    __gm__ void *mc2CcTiling = reinterpret_cast<__gm__ void *>(&tilingData->mc2CcTiling);
    auto context = AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();
    hccl.Init(context, mc2InitTiling);
    hccl.SetCcTiling(mc2CcTiling);

    if (tiling_data.allreduceAddRmsnormInfo.ppTilingData.isGatherAddOut) {
        AllreduceAddRmsnormAivKernel<DTYPE_LOCAL_PROJECTION, DTYPE_Y, true> op;
        op.Init(
            local_projection, residual, gamma, chain_state, y, add_out,
            next_chain_state, workspace, &tiling_data, hccl);
        op.Process(&tiling_data);
    } else {
        AllreduceAddRmsnormAivKernel<DTYPE_LOCAL_PROJECTION, DTYPE_Y, false> op;
        op.Init(
            local_projection, residual, gamma, chain_state, y, add_out,
            next_chain_state, workspace, &tiling_data, hccl);
        op.Process(&tiling_data);
    }
}
