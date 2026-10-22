"""HOST-only repeated ABI-check memo, strictly scoped to one task-list update.

The original checker still decides validity. No event, CANN call, graph flag,
validation counter, tensor value or Native file is changed. This prototype is
not installed by any production launcher and has no NPU performance claim.
"""
from contextvars import ContextVar
from functools import wraps


class ScopedTreeValidationMemo:
    def __init__(self, module):
        self.module = module
        self.original_validate = module._validate_tree_fia_mask
        self.runner_class = module.NativeACLGraphRunner
        self.original_update = self.runner_class._update_attention_task_list
        self.scope = ContextVar('specslo_tree_fia_host_validation', default=None)
        self.installed = False
        self.original_calls = self.memo_hits = self.task_lists = 0

        @wraps(self.original_validate)
        def validate(query_count, attention_mask, block_table,
                     actual_seq_lengths_q, actual_seq_lengths_kv, block_size):
            seen = self.scope.get()
            key = None
            if seen is not None:
                key = self._key(query_count, attention_mask, block_table,
                                actual_seq_lengths_q, actual_seq_lengths_kv, block_size)
                if key is not None and key in seen:
                    self.memo_hits += 1
                    return None
            self.original_calls += 1
            result = self.original_validate(query_count, attention_mask, block_table,
                                            actual_seq_lengths_q, actual_seq_lengths_kv, block_size)
            # Failed checks are never cached. Each update starts with a fresh
            # set, and a complete value-key catches mutable length changes.
            if seen is not None and key is not None:
                seen.add(key)
            return result

        @wraps(self.original_update)
        def update(runner, *args, **kwargs):
            token = self.scope.set(set())
            self.task_lists += 1
            try:
                return self.original_update(runner, *args, **kwargs)
            finally:
                self.scope.reset(token)

        self.validate = validate
        self.update = update

    def _key(self, query_count, mask, table, q, kv, block_size):
        torch = self.module.torch
        if (type(query_count) is not int or type(block_size) is not int
                or type(mask) is not torch.Tensor or type(table) is not torch.Tensor
                or type(q) not in (tuple, list) or type(kv) not in (tuple, list)):
            return None
        if not all(type(value) is int for value in (*q, *kv)):
            return None
        return (query_count, mask.ndim, mask.dtype, tuple(mask.shape), table.ndim,
                tuple(table.shape), tuple(q), tuple(kv), block_size)

    def install(self):
        if self.installed:
            raise RuntimeError('HOST memo already installed')
        if (self.module._validate_tree_fia_mask is not self.original_validate
                or self.runner_class._update_attention_task_list is not self.original_update):
            raise RuntimeError('Native methods changed before HOST memo installation')
        self.module._validate_tree_fia_mask = self.validate
        self.runner_class._update_attention_task_list = self.update
        self.installed = True
        return self

    def uninstall(self):
        if (not self.installed or self.module._validate_tree_fia_mask is not self.validate
                or self.runner_class._update_attention_task_list is not self.update):
            raise RuntimeError('HOST memo ownership changed before removal')
        self.module._validate_tree_fia_mask = self.original_validate
        self.runner_class._update_attention_task_list = self.original_update
        self.installed = False

    def metrics(self):
        return dict(task_lists=self.task_lists, original_checks=self.original_calls,
                    repeated_checks_reused=self.memo_hits,
                    live_scope=self.scope.get() is not None, installed=self.installed,
                    NPU_performance_proved=False, production_installed=False)
