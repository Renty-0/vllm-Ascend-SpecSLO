/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "register/op_def_registry.h"

namespace ge {
namespace {
constexpr uint32_t kResidualIndex = 1;
constexpr uint32_t kChainStateIndex = 3;

void CloneShape(const gert::Shape *src, gert::Shape *dst)
{
    dst->SetDimNum(src->GetDimNum());
    for (int32_t i = 0; i < src->GetDimNum(); ++i) {
        dst->SetDim(i, src->GetDim(i));
    }
}

graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *residual = context->GetInputShape(kResidualIndex);
    if (residual == nullptr || (residual->GetDimNum() != 2 && residual->GetDimNum() != 3)) {
        return GRAPH_FAILED;
    }
    CloneShape(residual, context->GetOutputShape(0));
    CloneShape(residual, context->GetOutputShape(1));
    const gert::Shape *chainState = context->GetInputShape(kChainStateIndex);
    if (chainState == nullptr || chainState->GetDimNum() != 2) {
        return GRAPH_FAILED;
    }
    CloneShape(chainState, context->GetOutputShape(2));
    return GRAPH_SUCCESS;
}

graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto dtype = context->GetInputDataType(kResidualIndex);
    context->SetOutputDataType(0, dtype);
    context->SetOutputDataType(1, dtype);
    context->SetOutputDataType(2, context->GetInputDataType(kChainStateIndex));
    return GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP(AllreduceAddRmsnorm).InferShape(InferShape).InferDataType(InferDataType);
}  // namespace ge
