#include "aclnn_allreduce_add_rmsnorm.h"

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak))
NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType type);

extern "C" {
extern aclnnStatus aclnnInnerAllreduceAddRmsnormGetWorkspaceSize(
    const aclTensor *localProjection,
    const aclTensor *residual,
    const aclTensor *gamma,
    const aclTensor *chainState,
    char *groupTp,
    int64_t tpRankSize,
    int64_t tpRankId,
    double epsilon,
    bool isGatherAddOut,
    bool flushChain,
    const aclTensor *y,
    const aclTensor *addOut,
    const aclTensor *nextChainState,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerAllreduceAddRmsnorm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

aclnnStatus aclnnAllreduceAddRmsnormGetWorkspaceSize(
    const aclTensor *localProjection,
    const aclTensor *residual,
    const aclTensor *gamma,
    const aclTensor *chainState,
    char *groupTp,
    int64_t tpRankSize,
    int64_t tpRankId,
    double epsilon,
    bool isGatherAddOut,
    bool flushChain,
    const aclTensor *y,
    const aclTensor *addOut,
    const aclTensor *nextChainState,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerAllreduceAddRmsnormGetWorkspaceSize(
        localProjection, residual, gamma, chainState, groupTp, tpRankSize,
        tpRankId, epsilon, isGatherAddOut, flushChain, y, addOut,
        nextChainState, workspaceSize, executor);
}

aclnnStatus aclnnAllreduceAddRmsnorm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    if (NnopbaseSetHcclServerType != nullptr) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerAllreduceAddRmsnorm(workspace, workspaceSize, executor, stream);
}
}  // extern "C"
