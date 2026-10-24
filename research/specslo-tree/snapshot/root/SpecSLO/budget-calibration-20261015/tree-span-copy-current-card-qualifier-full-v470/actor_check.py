"""Exercise the retained actual actor gate, without rewriting old evidence."""
import inspect
from pathlib import Path
import runpy


def check():
    path = Path(__file__).resolve().parent.parent / 'tree-current-card-component-binding-full-v465/check_HOST465.py'
    namespace = runpy.run_path(str(path), run_name='ROOT470_readonly_original_actor_check')
    source = inspect.getsource(namespace['main'])
    marker = '    # Actual factories compile;'
    if source.count(marker) != 1:
        raise RuntimeError('Exact original actor gate boundary required')
    source = source.split(marker)[0]
    source += '    return dict(positive=positive, negative=negative, actual_compiled_actor=True, complete_source_inverse=True)\n'
    exec(compile(source, str(path) + '::readonly-actual-actor', 'exec'), namespace)
    return namespace['main']()
