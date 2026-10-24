"""Execute the actual privately compiled component's device construction gate."""
from pathlib import Path
from types import FunctionType, SimpleNamespace
import ast
import copy
import hashlib
import inspect
import json
import os
import subprocess
import sys

P = Path(__file__).resolve().parent
B = P.parent
OLD = B / 'tree-current-card-rebind-full-v464'
shell = (P / 'startup465.sh').read_text()
line = next(x for x in shell.splitlines() if x.startswith('export PYTHONPATH='))
sys.path[:0] = line[len('export PYTHONPATH='):-len('${PYTHONPATH:-}')].split(':')
import base_component465
import extension465
import extension463
import runtime465
import guard465
import guard464


def cell(value):
    return (lambda: value).__closure__[0]


def main():
    component = base_component465.parent_component()
    actual = component.original_B4_run
    compiled = {}
    inherited_functions = actual.__globals__['functions']

    class Captured(Exception):
        pass

    def capture(path, namespace, **kwargs):
        if Path(path).name == 'full_process_v146.py':
            compiled['actor'] = namespace['run_actor']
            raise Captured()
        source = inherited_functions(path, namespace, **kwargs)
        if Path(path).name == 'actor_v146.py':
            compiled['source'] = source
        return source

    body = FunctionType(actual.__code__, dict(actual.__globals__, functions=capture),
                        actual.__name__, actual.__defaults__, actual.__closure__)
    try:
        body(None, None, None, None, None)
    except Captured:
        pass
    assert set(compiled) == {'actor', 'source'}
    changes = (('future_full_requires_21scope26phase_and_two65zero=True',
                'future_full_requires_standalone_scope_and_two65zero=True', 1),) + base_component465.ACTOR_CHANGES
    inverse = compiled['source']
    for before, after, count in reversed(changes):
        assert inverse.count(after) == count
        inverse = inverse.replace(after, before)
    original_actor = B / 'tree-high-normal-cap4-native-families-v148/actor_v146.py'
    assert inverse == original_actor.read_text()
    actor = compiled['actor']
    construction = next(x for x in actor.__code__.co_consts if hasattr(x, 'co_name') and x.co_name == 'construction')
    assert construction.co_freevars == ('engine',)
    ns = dict(actor.__globals__, validate_full_configuration=lambda e: dict(maxseq=64),
              native=lambda e: {}, inventory_record=lambda e: {})
    metadata = next(n.value for n in ast.walk(ast.parse(compiled['source']))
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'report' for t in n.targets))
    metadata_nodes = {kw.arg: kw.value for kw in metadata.keywords if kw.arg in ('physical_cards', 'physical_rank_card')}
    assert set(metadata_nodes) == {'physical_cards', 'physical_rank_card'}
    env_before = dict(os.environ)
    positive, negative = 0, 0
    try:
        for rank in range(4):
            engine = SimpleNamespace(rank=rank, local_rank=rank, is_draft=rank == 0,
                topology=SimpleNamespace(target_leader_rank=1),
                model=SimpleNamespace(layers=[None] * (28 if rank == 0 else 64)))
            fn = FunctionType(construction, ns, 'construction', closure=(cell(engine),))
            os.environ.update(SPECSLO_VISIBLE_DEVICES='0,1,2,7', ASCEND_RT_VISIBLE_DEVICES='0,1,2,3', LOCAL_RANK=str(rank))
            assert fn()['maxseq'] == 64
            fields = {key: eval(compile(ast.Expression(copy.deepcopy(node)), '<actual actor metadata>', 'eval'), {}, {'engine': engine})
                      for key, node in metadata_nodes.items()}
            assert fields == dict(physical_cards=[0,1,2,7], physical_rank_card=[0,1,2,7][rank])
            positive += 1
            mutations = [('SPECSLO_VISIBLE_DEVICES', '2,5,6,7'), ('ASCEND_RT_VISIBLE_DEVICES', '0,1,2,7'),
                         ('LOCAL_RANK', str((rank+1)%4)), ('engine_local', (rank+1)%4), ('leader', 0)]
            for key, value in mutations:
                before = dict(os.environ)
                if key == 'engine_local':
                    engine.local_rank = value
                elif key == 'leader':
                    engine.topology.target_leader_rank = value
                else:
                    os.environ[key] = value
                try:
                    fn()
                except RuntimeError as error:
                    assert 'actual physical visibility/local-rank/target-leader mapping' in str(error)
                    negative += 1
                else:
                    raise AssertionError('Wrong actual device metadata accepted')
                os.environ.clear(); os.environ.update(before)
                engine.local_rank = rank; engine.topology.target_leader_rank = 1
    finally:
        os.environ.clear(); os.environ.update(env_before)

    # Actual factories compile; only base qualifier route differs, preserving all added math gates.
    projected = extension465.projected_qualifier()
    assert 'base_component465' in projected.__code__.co_names
    assert callable(extension465.qualifier())
    assert callable(base_component465.construct209())
    for name in ('prepare_install', 'install_entry', 'after_install', 'cache_snapshot', 'bucket_snapshot',
                 'metadata_snapshot', 'refill_snapshot', 'preflight_snapshot', 'postfull_AR', 'snapshot'):
        assert getattr(extension465, name) is getattr(extension463, name)
    source = runtime465.source()
    assert source.replace("extension_module='extension465'", "extension_module='extension463'").replace(
        "require(extension_module=='extension465'", "require(extension_module=='extension463'") == runtime465.original.source()
    previous, current = guard464.execution_namespace(), guard465.execution_namespace()
    a, z = previous['execute_locked'].__code__, current['execute_locked'].__code__
    assert a.co_code == z.co_code and a.co_names == z.co_names and a.co_varnames == z.co_varnames
    assert [(x,y) for x,y in zip(a.co_consts,z.co_consts) if x!=y] == [
        ('ROOT.V464.current-card-credit-cost-full-own-AR-supervisor.v1', 'ROOT.V465.current-component-credit-cost-full-own-AR-supervisor.v1')]
    assert current['actual_audit'] is previous['actual_audit']
    assert current['SCOPE']['global_active_limit'] == 64
    for key in ('MAX_SECONDS','MAX_TAG_BYTES','MAX_AGGREGATE_BYTES'):
        assert current[key] == previous[key]
    for path in P.glob('*.sh'):
        subprocess.run(['bash','-n',str(path)],check=True)
    for path in P.glob('*.py'):
        ast.parse(path.read_bytes(), filename=str(path))
    result = dict(status='PASS_HOST465_ACTUAL_COMPONENT_CURRENT_CARDS',
        actual_privately_compiled_actor_checked=True, actor_complete_source_inverse=True,
        actual_nested_construction_positive=positive, actual_nested_construction_negative=negative,
        actual_rank_metadata_checked=True, actual_qualifier_factories_compiled=True,
        ten_original_numerical_callbacks_identical=True, original_worker_complete_source_inverse=True,
        supervisor_bytecode_identical=True, physical_cards=[0,1,2,7], global_active_limit=64,
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in P.iterdir()
                 if p.is_file() and p.suffix in ('.py','.sh','.json') and not p.name.startswith(('HOST','FREEZE','SOURCE'))},
        NPU_used=False, NN_permission=False)
    with (P/'HOST465_RESULT.json').open('x') as stream:
        json.dump(result,stream,indent=2); stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sources'}))


if __name__ == '__main__':
    main()
