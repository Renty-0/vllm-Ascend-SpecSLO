add_op_to_compiled_list()

set(_MATMUL_ALLREDUCE_INC_OPTS)
if (EXISTS ${CMAKE_SOURCE_DIR}/third_party/catlass/include)
    list(APPEND _MATMUL_ALLREDUCE_INC_OPTS -I${CMAKE_SOURCE_DIR}/third_party/catlass/include)
else()
    message(FATAL_ERROR "dependency catlass is missing, you can fetch it by running 'git submodule update --init --recursive'")
endif()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        matmul_allreduce_add_rmsnorm_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME MatmulAllreduceAddRmsnorm
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
        -DCATLASS_ARCH=2201
        ${_MATMUL_ALLREDUCE_INC_OPTS}
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE matmul_allreduce_add_rmsnorm ACLNNTYPE aclnn_inner)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
