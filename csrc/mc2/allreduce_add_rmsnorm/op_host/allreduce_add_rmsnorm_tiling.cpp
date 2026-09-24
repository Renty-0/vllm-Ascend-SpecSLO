/*
 * Experimental SpecSLO TP3 AIV-only all-reduce + AddRMSNorm tiler.
 *
 * This candidate intentionally accepts the local projection produced by the
 * native F.linear path.  It contains no Cube/MatMul tiling and is fail-closed
 * to the Qwen3-32B TP3 verify window qualified by the surrounding harness.
 */

#include <cstdint>
#include <string>

#include "error/ops_error.h"
#include "graph/utils/type_utils.h"
#include "log/ops_log.h"
#include "register/op_def_registry.h"
#include "tiling/hccl/hccl_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/allreduce_add_rmsnorm_tiling.h"

namespace {
enum AttrIndex : uint32_t {
    ATTR_GROUP_TP = 0,
    ATTR_RANK_SIZE,
    ATTR_RANK_ID,
    ATTR_EPSILON,
    ATTR_GATHER_ADD_OUT,
    ATTR_FLUSH_CHAIN,
};

constexpr int32_t kTpSize = 3;
constexpr int32_t kMaxM = 160;
constexpr int32_t kHiddenSize = 5120;
constexpr int32_t kChainStateRows = 64;
constexpr int32_t kChainStateWords = 4;
// The TP3 mailbox protocol performs two remote polls per launched AIV.  A
// smaller cohort can reduce HCCS control-traffic contention while retaining
// enough row-level parallelism for the qualified M<=160 verification window.
constexpr uint32_t kMaxAivBlocks = 24;
constexpr float kRequiredEpsilon = 1.0e-6F;
constexpr float kEpsilonTolerance = 1.0e-12F;
constexpr int32_t kUbMoveElements = 20480;
constexpr uint32_t kSystemWorkspaceFallback = 16U * 1024U * 1024U;
constexpr uint32_t kOpTypeAllToAll = 8;

bool SameShape(const gert::Shape &a, const gert::Shape &b)
{
    if (a.GetDimNum() != b.GetDimNum()) {
        return false;
    }
    for (int32_t i = 0; i < a.GetDimNum(); ++i) {
        if (a.GetDim(i) != b.GetDim(i)) {
            return false;
        }
    }
    return true;
}

ge::graphStatus FillTiling(gert::TilingContext *context, AllreduceAddRmsnormTilingData &tiling)
{
    const char *nodeName = context->GetNodeName();
    auto *attrs = context->GetAttrs();
    auto *projection = context->GetInputTensor(0);
    auto *residual = context->GetInputTensor(1);
    auto *gamma = context->GetInputTensor(2);
    auto *chainState = context->GetInputTensor(3);
    OPS_ERR_IF(
        attrs == nullptr || projection == nullptr || residual == nullptr ||
        gamma == nullptr || chainState == nullptr,
        OPS_LOG_E(nodeName, "chained epilogue requires four non-null inputs and attributes."),
        return ge::GRAPH_FAILED);

    const auto *rankSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_RANK_SIZE);
    const auto *rankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_RANK_ID);
    const auto *epsilonPtr = attrs->GetAttrPointer<float>(ATTR_EPSILON);
    const auto *gatherPtr = attrs->GetAttrPointer<bool>(ATTR_GATHER_ADD_OUT);
    const auto *flushChainPtr = attrs->GetAttrPointer<bool>(ATTR_FLUSH_CHAIN);
    OPS_ERR_IF(
        rankSizePtr == nullptr || rankIdPtr == nullptr || epsilonPtr == nullptr ||
        gatherPtr == nullptr || flushChainPtr == nullptr,
        OPS_LOG_E(nodeName, "chained epilogue required attributes are missing."),
        return ge::GRAPH_FAILED);

    const auto &projectionShape = projection->GetOriginShape();
    const int32_t dims = projectionShape.GetDimNum();
    int64_t m = 0;
    int64_t n = 0;
    if (dims == 2) {
        m = projectionShape.GetDim(0);
        n = projectionShape.GetDim(1);
    } else if (dims == 3) {
        m = projectionShape.GetDim(0) * projectionShape.GetDim(1);
        n = projectionShape.GetDim(2);
    } else {
        OPS_LOG_E(nodeName, "v126 local_projection must be rank 2 or 3.");
        return ge::GRAPH_FAILED;
    }

    const auto &gammaShape = gamma->GetOriginShape();
    const auto &chainStateShape = chainState->GetOriginShape();
    const auto projectionFormat = static_cast<ge::Format>(
        ge::GetPrimaryFormat(context->GetInputDesc(0)->GetStorageFormat()));
    const auto residualFormat = static_cast<ge::Format>(
        ge::GetPrimaryFormat(context->GetInputDesc(1)->GetStorageFormat()));
    const auto gammaFormat = static_cast<ge::Format>(
        ge::GetPrimaryFormat(context->GetInputDesc(2)->GetStorageFormat()));
    const auto chainStateFormat = static_cast<ge::Format>(
        ge::GetPrimaryFormat(context->GetInputDesc(3)->GetStorageFormat()));
    const int64_t rankSize = *rankSizePtr;
    const int64_t rankId = *rankIdPtr;

    const bool invalid =
        rankSize != kTpSize || rankId < 0 || rankId >= kTpSize ||
        m <= 0 || m > kMaxM || n != kHiddenSize ||
        projection->GetDataType() != ge::DT_BF16 ||
        residual->GetDataType() != ge::DT_BF16 || gamma->GetDataType() != ge::DT_BF16 ||
        chainState->GetDataType() != ge::DT_INT64 ||
        projectionFormat != ge::FORMAT_ND || residualFormat != ge::FORMAT_ND ||
        gammaFormat != ge::FORMAT_ND || chainStateFormat != ge::FORMAT_ND ||
        !SameShape(projectionShape, residual->GetOriginShape()) ||
        gammaShape.GetDimNum() != 1 || gammaShape.GetDim(0) != n ||
        chainStateShape.GetDimNum() != 2 ||
        chainStateShape.GetDim(0) != kChainStateRows ||
        chainStateShape.GetDim(1) != kChainStateWords ||
        *epsilonPtr < kRequiredEpsilon - kEpsilonTolerance ||
        *epsilonPtr > kRequiredEpsilon + kEpsilonTolerance;
    OPS_ERR_IF(invalid,
        OPS_LOG_E(nodeName,
            "v126 admission details: rankSize=%ld rankId=%ld dims=%d m=%ld n=%ld "
            "projectionDtype=%d residualDtype=%d gammaDtype=%d "
            "chainStateDtype=%d projectionFormat=%d residualFormat=%d "
            "gammaFormat=%d chainStateFormat=%d sameShape=%d gammaDims=%d "
            "gamma0=%ld chainStateDims=%d chainState0=%ld chainState1=%ld epsilon=%.12g.",
            rankSize, rankId, dims, m, n,
            static_cast<int32_t>(projection->GetDataType()),
            static_cast<int32_t>(residual->GetDataType()),
            static_cast<int32_t>(gamma->GetDataType()),
            static_cast<int32_t>(chainState->GetDataType()),
            static_cast<int32_t>(projectionFormat),
            static_cast<int32_t>(residualFormat),
            static_cast<int32_t>(gammaFormat),
            static_cast<int32_t>(chainStateFormat),
            static_cast<int32_t>(SameShape(projectionShape, residual->GetOriginShape())),
            gammaShape.GetDimNum(),
            gammaShape.GetDimNum() > 0 ? gammaShape.GetDim(0) : -1,
            chainStateShape.GetDimNum(),
            chainStateShape.GetDimNum() > 0 ? chainStateShape.GetDim(0) : -1,
            chainStateShape.GetDimNum() > 1 ? chainStateShape.GetDim(1) : -1,
            static_cast<double>(*epsilonPtr)),
        return ge::GRAPH_FAILED);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aivNum = platform.GetCoreNumAiv();
    OPS_ERR_IF(aivNum == 0,
        OPS_LOG_E(nodeName, "v126 cannot launch without AIV cores."),
        return ge::GRAPH_FAILED);

    uint32_t launchedAiv = aivNum;
    if (launchedAiv > static_cast<uint32_t>(m)) {
        launchedAiv = static_cast<uint32_t>(m);
    }
    if (launchedAiv > kMaxAivBlocks) {
        launchedAiv = kMaxAivBlocks;
    }

    auto &info = tiling.allreduceAddRmsnormInfo;
    auto &pp = info.ppTilingData;
    auto &comm = info.commTilingData;
    auto &quant = info.quantInfo;
    pp.opShape.batchSize = 1;
    pp.opShape.m = static_cast<int32_t>(m);
    pp.opShape.k = static_cast<int32_t>(n);
    pp.opShape.n = static_cast<int32_t>(n);
    pp.m0 = 128;
    pp.k0 = 256;
    pp.n0 = 256;
    pp.mLoop = (pp.opShape.m + pp.m0 - 1) / pp.m0;
    pp.kLoop = (pp.opShape.k + pp.k0 - 1) / pp.k0;
    pp.nLoop = (pp.opShape.n + pp.n0 - 1) / pp.n0;
    pp.coreLoop = pp.mLoop * pp.nLoop;
    pp.swizzlCount = 4;
    pp.swizzlDirect = 1;
    pp.blockDim = static_cast<int32_t>(launchedAiv);
    pp.flushChain = *flushChainPtr ? 1 : 0;
    pp.epsilon = 1.0e-6F;
    pp.isGatherAddOut = *gatherPtr;
    pp.projectionOnly = false;

    comm.rank = static_cast<int32_t>(rankId);
    comm.rankSize = static_cast<int32_t>(rankSize);
    comm.ubMoveNum = kUbMoveElements;
    comm.pValue = 1;
    comm.commNpuSplit = kTpSize;
    comm.commDataSplit = 1;
    comm.commDirect = 0;
    comm.withSerialMode = 0;
    comm.tag = 0;
    comm.write2OtherRank = 0;
    comm.is91093 = 0;

    quant.dequantGranularity = QuantGranularity::QUANT_GRANULARITY_UNDEFINED;
    quant.dequantGroupSize = -1;
    quant.quantGranularity = QuantGranularity::QUANT_GRANULARITY_UNDEFINED;
    quant.quantGroupSize = -1;

    // Each block owns an independent row chunk and rendezvous only with the
    // matching block on the other TP ranks.  Keeping blockDim no larger than
    // the physical AIV cohort avoids oversubscription without imposing batch
    // scheduling on native F.linear or the surrounding ACLGraph.
    context->SetBlockDim(launchedAiv);
    return ge::GRAPH_SUCCESS;
}

void SetHcommCfg(AllreduceAddRmsnormTilingData &tiling)
{
    AscendC::Mc2CcTilingConfig config(
        "hcomms", kOpTypeAllToAll, "AlltoAll=level0:fullmesh");
    config.GetTiling(tiling.mc2InitTiling);
    config.GetTiling(tiling.mc2CcTiling);
}

ge::graphStatus TilingImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    auto *tiling = context->GetTilingData<AllreduceAddRmsnormTilingData>();
    OPS_ERR_IF(tiling == nullptr,
        OPS_LOG_E(nodeName, "v126 tiling buffer is null."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(FillTiling(context, *tiling) != ge::GRAPH_SUCCESS,
        OPS_LOG_E(nodeName, "v126 input or attribute admission failed."),
        return ge::GRAPH_FAILED);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto *workspace = context->GetWorkspaceSizes(1);
    OPS_ERR_IF(workspace == nullptr,
        OPS_LOG_E(nodeName, "v126 workspace descriptor is null."),
        return ge::GRAPH_FAILED);
    const size_t systemWorkspace = static_cast<size_t>(platform.GetLibApiWorkSpaceSize());
    workspace[0] = systemWorkspace == 0 ? kSystemWorkspaceFallback : systemWorkspace;
    SetHcommCfg(*tiling);
    return ge::GRAPH_SUCCESS;
}

struct AllreduceAddRmsnormCompileInfo {};
ge::graphStatus Parse(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(AllreduceAddRmsnorm)
    .Tiling(TilingImpl)
    .TilingParse<AllreduceAddRmsnormCompileInfo>(Parse);
