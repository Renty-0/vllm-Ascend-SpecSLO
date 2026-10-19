"""Unchanged336 math/cost; bind actual frontend and retain full local errors."""
import inspect
import json
from pathlib import Path
import traceback
import probe336 as original


def projected_run():
    source = inspect.getsource(original.run)
    changes = (
        ('from metadata336 import install, ensure_zero', 'from metadata338 import install, ensure_zero'),
        ("    model_metrics_before = engine.graph_metrics()",
         "    stage('ROOT338/retain-independent-bank-before-math', lambda: _save_bank338(output,engine.rank,bank['evidence']))\n    model_metrics_before = engine.graph_metrics()"))
    projected = source
    for before, after in changes:
        if projected.count(before) != 1:
            raise RuntimeError('ROOT338 exact non-numeric projection anchor '+before)
        projected = projected.replace(before, after)
    inverse = projected
    for before, after in reversed(changes):
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise RuntimeError('ROOT338 entire finite math/cost source inverse')
    namespace = dict(vars(original), _save_bank338=save_bank)
    exec(compile(projected, __file__+'::unchanged336-run-with-actual-builder-and-error-records', 'exec'), namespace)
    return namespace['run']


def save_bank(output, rank, evidence):
    with (Path(output)/f'rank{rank}-independent-bank-before-math338.json').open('x') as saved:
        saved.write(json.dumps(dict(rank=rank, evidence=evidence, complete_model_case=False,
                                   performance=False), indent=2)+'\n')


def run(engine, native, stage, output, fixture):
    failures = 0
    def retained_stage(label, function):
        def execute():
            nonlocal failures
            try:
                return function()
            except BaseException:
                failures += 1
                path = Path(output)/f'rank{engine.rank}-local-stage-failure338-{failures}.json'
                with path.open('x') as saved:
                    saved.write(json.dumps(dict(rank=engine.rank, label=label,
                        error=traceback.format_exc(), performance=False), indent=2)+'\n')
                raise
        return stage(label, execute)
    return projected_run()(engine, native, retained_stage, output, fixture)
