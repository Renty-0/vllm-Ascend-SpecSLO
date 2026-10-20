"""Reuse causal-prefill attention task buffers on the serial target stream.

Eager execution stays the numerical oracle. During capture, explicit graph
copies feed shared query/output storage; all original attention parameters,
task events, native replay validation and restoration checks remain in place.
"""
_SCOPE = None


def install(graph_module, model_module, maximum_rows=1024):
    import torch
    original = graph_module.run_native_fused_infer_attention
    if model_module.run_native_fused_infer_attention is not original:
        raise RuntimeError('Unexpected native attention function binding')
    runner_cls = graph_module.NativeACLGraphRunner
    hidden = runner_cls.run_tree_hidden

    def scoped(runner, *args, **kwargs):
        global _SCOPE
        if getattr(runner, '_dynamic_exact_prefill_expected_rows', None) is None:
            return hidden(runner, *args, **kwargs)
        previous = _SCOPE
        if previous is not None:
            raise RuntimeError('Prefill arena does not allow concurrent/nested graph execution')
        if not hasattr(runner, '_dynamic_prefill_arenas'):
            runner._dynamic_prefill_arenas = {}
        _SCOPE = runner._dynamic_prefill_arenas
        try:
            return hidden(runner, *args, **kwargs)
        finally:
            _SCOPE = previous

    def attention(**kwargs):
        if _SCOPE is None or kwargs.get('tree_attention', False):
            return original(**kwargs)
        query, output = kwargs['query'], kwargs['output']
        if (query.shape != output.shape or query.dtype != output.dtype
                or query.ndim != 3 or not 1 <= query.shape[0] <= maximum_rows):
            raise RuntimeError('Prefill attention is outside the arena layout/capacity contract')
        key = (str(query.device), query.dtype, tuple(query.shape[1:]))
        buffers = _SCOPE.get(key)
        capturing = graph_module._CAPTURED_TASKS is not None
        if buffers is None:
            if capturing:
                raise RuntimeError('Prefill arena must be allocated during eager qualification')
            shape = (maximum_rows, *query.shape[1:])
            buffers = (torch.empty(shape, dtype=query.dtype, device=query.device),
                       torch.empty(shape, dtype=output.dtype, device=output.device))
            _SCOPE[key] = buffers
        if not capturing:
            return original(**kwargs)
        rows = int(query.shape[0])
        shared_query, shared_output = (buffer[:rows] for buffer in buffers)
        shared_query.copy_(query)
        original(**(kwargs | {'query': shared_query, 'output': shared_output}))
        output.copy_(shared_output)

    graph_module.run_native_fused_infer_attention = attention
    model_module.run_native_fused_infer_attention = attention
    runner_cls.run_tree_hidden = scoped


def retained_task_storage(runner):
    """Unique device storage retained by FIA task query/output references."""
    storages = {}
    tasks = 0
    for entry in runner.target_entries.values():
        for task in entry.tasks:
            tasks += 1
            for name in ('query', 'output'):
                value = getattr(task, name)
                storage = value.untyped_storage()
                storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
    return {'fia_task_count': tasks, 'task_query_output_unique_storages': len(storages),
            'task_query_output_storage_bytes': sum(storages.values())}
