"""Protect the incumbent plus every nonselected candidate/reference owner."""
import ast
import inspect

import subset209 as original


def guard_rest(engine, bank, selected):
    selected_set = set(selected.ids)
    histories = {i:bank.contexts[i][:-1] for i in range(bank.n) if i not in selected_set}
    histories.update({i+bank.n:bank.reference_contexts[i][:-1]
                      for i in range(bank.n) if i not in selected_set})
    histories[bank.incumbent] = bank.incumbent_tokens
    return original.prefix_evidence(engine, histories)


def namespace():
    source = inspect.getsource(original.prepare)
    before, after = '(1 if bank.n==2 else 16)', 'min(16,(bank.n+1)//2)'
    if source.count(before) != 1:
        raise RuntimeError('one original selected-home occupancy predicate')
    projected = source.replace(before, after)
    if projected.replace(after, before) != source:
        raise RuntimeError('whole original planner preparation source inverse')
    value = dict(vars(original), guard_rest=guard_rest)
    exec(compile(ast.parse(projected), __file__+'::original-small-home-selection', 'exec'), value)
    # Rebind the unchanged finish body to exactly the same protection function.
    source_finish = inspect.getsource(original.finish)
    exec(compile(ast.parse(source_finish), __file__+'::original-finish-with-incumbent-protection', 'exec'), value)
    return value


def prepare(*args, **kwargs): return namespace()['prepare'](*args, **kwargs)
def finish(*args, **kwargs): return namespace()['finish'](*args, **kwargs)
