"""Opt-in replay-first for the actual tree target entry, preserving validation."""
import inspect
import textwrap


def submit(runner, entry, lengths_unchanged, current_stream):
    """Release each captured attention gate only after its metadata is ready."""
    if not entry.tasks:
        raise RuntimeError('Tree replay-first requires captured attention tasks')
    # Tree entries predate the generic entry's copy-event field. Reuse the
    # original validated event factory after adding that entry-local field.
    if not hasattr(entry, 'replay_first_copy_done_event'):
        entry.replay_first_copy_done_event = None
    copy_done = runner._generic_replay_first_copy_event(entry)
    try:
        # This event precedes graph submission on the same current stream.
        # It covers both previous graph completion/reset and new input copies.
        copy_done.record(current_stream)
        runner.update_stream.wait_event(copy_done)
    except Exception as exc:
        raise RuntimeError('Tree replay-first input dependency failed before submission') from exc
    try:
        entry.graph.replay()
    except Exception as exc:
        raise RuntimeError('Tree replay-first graph submission failed') from exc
    try:
        if lengths_unchanged:
            for task in entry.tasks:
                task.event.record(runner.update_stream)
            runner.task_update_skip_replay_count += 1
        else:
            runner._update_target_attention_tasks(entry)
            runner._record_task_update('target', len(entry.tasks))
    except Exception as exc:
        # Submitted graphs can already have written KV or be waiting on gates.
        # Returning an eager fallback after a partial update is unsafe.
        raise RuntimeError('Tree replay-first update failed after submission; no safe fallback') from exc
    runner._dynamic_tree_target_replay_first_calls = (
        getattr(runner, '_dynamic_tree_target_replay_first_calls', 0) + 1)
    runner._dynamic_tree_target_replay_first_updates = (
        getattr(runner, '_dynamic_tree_target_replay_first_updates', 0)
        + int(not lengths_unchanged))


def patched_source(original):
    source = textwrap.dedent(original)
    start = source.index('    if task_lengths_unchanged:\n', source.index('    self._copy_target_inputs('))
    stop = source.index('    self.replay_count += 1\n', start)
    ordinary = source[start:stop]
    assert ordinary.count('entry.graph.replay()') == 1
    replacement = (
        "    if getattr(self, '_dynamic_tree_target_replay_first', False):\n"
        '        _dynamic_tree_replay_submit(self, entry, task_lengths_unchanged, current_stream)\n'
        '    else:\n' + textwrap.indent(ordinary, '    '))
    return source[:start] + replacement + source[stop:]


def install(graph_module):
    cls = graph_module.NativeACLGraphRunner
    source = patched_source(inspect.getsource(cls._run_target))
    namespace = dict(vars(graph_module), _dynamic_tree_replay_submit=submit)
    exec(compile(source, __file__ + ':_run_target', 'exec'), namespace)
    cls._run_target = namespace['_run_target']
    metrics = cls.graph_execution_metrics

    def measured(runner):
        return metrics(runner) | {
            'dynamic_tree_target_replay_first_enabled': bool(
                getattr(runner, '_dynamic_tree_target_replay_first', False)),
            'dynamic_tree_target_replay_first_calls': int(
                getattr(runner, '_dynamic_tree_target_replay_first_calls', 0)),
            'dynamic_tree_target_replay_first_updates': int(
                getattr(runner, '_dynamic_tree_target_replay_first_updates', 0))}

    cls.graph_execution_metrics = measured
