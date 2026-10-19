"""Extend the original PA updater only for explicit encoded top2 chains."""
import inspect
import textwrap


def patched_source(original):
    source=textwrap.dedent(original)
    token='DRAFT_PA_PARALLEL_FULL_CHAIN_STEPS'
    assert source.count(token)==2
    source=source.replace(token,'chain_steps')
    before='    expected_step_count = chain_steps + int(final_kv_only)\n'
    assert source.count(before)==1
    return source.replace(before,
        '    chain_steps = DRAFT_PA_PARALLEL_FULL_CHAIN_STEPS\n'
        '    encoded = getattr(entry, "output", None)\n'
        '    if (getattr(self, "_dynamic_tree_top2", False) and torch.is_tensor(encoded)\n'
        '            and encoded.ndim==3 and encoded.shape[2]==2 and encoded.dtype==torch.long\n'
        '            and 1<=encoded.shape[1]<=8):\n'
        '        chain_steps = int(encoded.shape[1])\n'+before)


def install(graph_module):
    cls=graph_module.NativeACLGraphRunner
    source=patched_source(inspect.getsource(cls._try_parallel_draft_paged_attention_task_update))
    namespace=dict(vars(graph_module))
    exec(compile(source,__file__+':parallel_steps','exec'),namespace)
    cls._try_parallel_draft_paged_attention_task_update=namespace['_try_parallel_draft_paged_attention_task_update']
