"""Qualify changed-input tree verification, restoring real KV before return."""
from collections import Counter


def install(native):
    runner_cls = native.NativeACLGraphRunner
    record = runner_cls._record_execution

    def tracked(runner, kind, execution, output, *, entry_key=None, token_rows=()):
        if kind == 'target':
            runner._dynamic_verification_last_key = entry_key
        return record(runner, kind, execution, output,
                      entry_key=entry_key, token_rows=token_rows)

    runner_cls._record_execution = tracked
    cls = native.NativePearlEngine
    original = cls.target_tree_forward

    def qualified(engine, plans, roots, rows, *args, **kwargs):
        result = original(engine, plans, roots, rows, *args, **kwargs)
        if engine.is_draft or engine.graph_runner.graph_cache_sealed:
            return result
        runner = engine.graph_runner
        key = getattr(runner, '_dynamic_verification_last_key', None)
        entry = runner.target_entries.get(key)
        if entry is None or entry.runtime_validated:
            return result
        # No commit happens in this helper. The alternate forward changes the
        # root input, invokes the ORIGINAL native changed-input validation, and
        # is followed immediately by an actual-input forward restoring every
        # root/candidate KV write before any target verdict is consumed.
        expected = result['target_query_token_ids'].clone()
        alternate = [(int(token) + 1) % engine.draft_vocab_size for token in roots]
        original(engine, plans, alternate, rows, *args, **kwargs)
        restored = original(engine, plans, roots, rows, *args, **kwargs)
        import torch
        if not entry.runtime_validated:
            raise RuntimeError('Tree verification changed-input qualification did not complete')
        if not torch.equal(expected, restored['target_query_token_ids']):
            raise RuntimeError('Tree verification qualification did not restore actual predictions')
        counts = getattr(engine, '_dynamic_verification_qualification_counts', None)
        if counts is None:
            counts = engine._dynamic_verification_qualification_counts = Counter()
        counts['qualified_entries'] += 1
        counts['untimed_extra_forwards'] += 2
        return restored

    cls.target_tree_forward = qualified
