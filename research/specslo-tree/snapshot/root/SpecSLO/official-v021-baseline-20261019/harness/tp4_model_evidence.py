"""Read worker model selection outside timed intervals; never install counters."""
def describe_worker(worker):
    from collections import Counter
    from vllm.config import set_current_vllm_config
    from vllm_ascend.ascend_config import get_ascend_config
    import vllm
    import vllm_ascend
    with set_current_vllm_config(worker.vllm_config):
        enabled = get_ascend_config().enable_matmul_allreduce
    histogram = Counter()
    hcoms = set()
    prefixes = Counter()
    for name, layer in worker.model_runner.model.named_modules():
        operation = getattr(layer, "custom_op", None)
        if operation is not None and type(operation).__name__ == "MatmulAllreduceRowParallelOp":
            histogram[str((tuple(layer.weight.shape), str(layer.weight.dtype), operation.tp_size))] += 1
            hcoms.add(operation.hcomm_info)
            prefixes[name.rsplit(".", 1)[-1]] += 1
    return {"rank": worker.rank, "enabled": enabled, "selected_modules": sum(histogram.values()),
            "weight_shape_dtype_tp_histogram": dict(histogram), "projection_counts": dict(prefixes),
            "hcoms": sorted(hcoms), "quantization": worker.vllm_config.model_config.quantization,
            "vllm_source": vllm.__file__, "vllm_ascend_source": vllm_ascend.__file__,
            "resolved_graph_mode": str(worker.vllm_config.compilation_config.cudagraph_mode),
            "resolved_async_scheduling": worker.vllm_config.scheduler_config.async_scheduling,
            "scope": "Resolved configuration and selected actual model classes; no per-device kernel count claimed"}

class TP4EvidenceExtension:
    def specslo_tp4_model_evidence(self):
        return describe_worker(self)
