/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "register/op_def_registry.h"

namespace ops {
class AllreduceAddRmsnorm : public OpDef {
public:
    explicit AllreduceAddRmsnorm(const char *name) : OpDef(name)
    {
        this->Input("local_projection")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("residual")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("chain_state")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("add_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("next_chain_state")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("group_tp").String();
        this->Attr("tp_rank_size").Int();
        this->Attr("tp_rank_id").Int();
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1.0e-6F);
        this->Attr("is_gather_add_out").AttrType(OPTIONAL).Bool(true);
        this->Attr("flush_chain").AttrType(OPTIONAL).Bool(true);

        this->MC2().HcclGroup({"group_tp"});
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AllreduceAddRmsnorm);
}  // namespace ops
