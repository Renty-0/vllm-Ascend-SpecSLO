#ifndef ACLNN_ALLREDUCE_ADD_RMSNORM
#define ACLNN_ALLREDUCE_ADD_RMSNORM

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnAllreduceAddRmsnormGetWorkspaceSize(
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

__attribute__((visibility("default"))) aclnnStatus aclnnAllreduceAddRmsnorm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
