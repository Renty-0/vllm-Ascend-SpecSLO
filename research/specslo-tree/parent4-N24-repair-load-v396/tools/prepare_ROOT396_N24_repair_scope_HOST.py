"""Prepare a finite N24/E4/R0..24 protocol successor; this never runs an NPU.

The source-only393 role projection is copied, not changed in place. The new
runtime retains original173 qualification and the independently checked391
M120 backend. No source descriptor supplies online performance permission.
"""
import ast
import hashlib
import json
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
PREVIOUS = B / 'tree-parent4-normal24-draft-shape-host-v393'
P = B / 'tree-parent4-normal24-repair-load-protocol-v396'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def changed(source, edits):
    result = source
    for before, after, count in edits:
        assert result.count(before) == count, (before, result.count(before))
        result = result.replace(before, after)
    inverse = result
    for before, after, _ in reversed(edits):
        inverse = inverse.replace(after, before)
    assert inverse == source
    return result


def write(name, source):
    if name.endswith('.py'):
        ast.parse(source, filename=name)
    with (P / name).open('x') as stream:
        stream.write(source)


def main():
    assert not P.exists()
    P.mkdir()
    projection = {}
    for name in ('scope393.py', 'draft393.py', 'actor393.py', 'protocol393.py',
                 'next393.py', 'budget393.py', 'provider393.py', 'binding393.py', 'live393.py'):
        source = (PREVIOUS / name).read_text()
        edits = [('393', '396', source.count('393'))]
        result = changed(source, edits)
        extra = []
        if name == 'scope393.py':
            extra = [
                ('1<=self.eager<=8', 'self.eager==4', 1),
                ('self.repair_count in (0,1)', '0<=self.repair_count<=24 ', 1),
                ('N24/P24 E1..8/R0..1', 'N24/P24 E4/R0..24', 1),
                ('Case(e,r)for e in range(1,9)for r in (0,1)',
                 'Case(4,r)for r in range(25)', 1),
                ('source_only=True', 'source_only=False,finite_component_only=True', 1),
            ]
        elif name == 'live393.py':
            extra = [
                ('frozenset((24,e)for e in range(1,9)),frozenset((e,1)for e in range(1,9))',
                 'frozenset({(24,4)}),frozenset((4,r)for r in range(1,25))', 1),
                ('N24/E1..8/single repair/M28or32', 'N24/E4/R0..24/M28', 1),
            ]
        elif name == 'next393.py':
            # Correct descriptions only; all model/reference operations stay identical.
            extra = [('parent3', 'parent4', result.count('parent3')),
                     ('child3', 'child4', result.count('child3'))]
        result = changed(result, extra)
        output = name.replace('393', '396')
        write(output, result)
        projection[output] = dict(source=str(PREVIOUS / name), source_sha256=digest(PREVIOUS / name),
                                  edits=edits + extra, inverse_verified=True)

    source = (B / 'tree-parent4-current-M120-target-protocol-v391/runtime391.py').read_text()
    edits = [('391', '396', source.count('391'))]
    # Only callback names change. The original worker, baseline, shutdown and
    # result-writing behavior remain the same projected original199 functions.
    runtime = changed(source, edits)
    write('runtime396.py', runtime)
    projection['runtime396.py'] = dict(
        source=str(B / 'tree-parent4-current-M120-target-protocol-v391/runtime391.py'),
        edits=edits, inverse_verified=True)

    extension = '''"""Own canonical FastV5 factory; fresh original baseline and finite M120 proof."""
import ast
import importlib
from pathlib import Path
import sys
import extension199 as original
import extension391


def prepare_install(output):
    source=Path(original.__file__).read_text()
    before='from live199 import prepare_helpers'
    after='from live396 import prepare_helpers'
    if source.count(before)!=1:raise RuntimeError('ROOT396 exact prepare helper anchor')
    projected=source.replace(before,after)
    if projected.replace(after,before)!=source:raise RuntimeError('ROOT396 prepare inverse')
    node=next(n for n in ast.parse(projected).body if isinstance(n,ast.FunctionDef)and n.name=='prepare_install')
    namespace=dict(vars(original))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
        __file__+'::original199-prepare-own-canonical-source','exec'),namespace)
    return namespace['prepare_install'](output)


after_install=extension391.after_install


def loaded_origins():
    from base199 import BASE,checked,sha,require
    from import_origins_v132 import KEYS
    from check396 import ready
    pins=ready()['files'];here=Path(__file__).resolve().parent;values={}
    for name in KEYS:
        module=importlib.import_module(name)
        path=Path(vars(module).get('__file__','')).resolve()
        expected=here/'fast396.generated.py'if name=='fast_tree_draft_v5'else BASE/(name+'.py')
        require(path==expected,'actual canonical396 origin '+name)
        if name=='fast_tree_draft_v5':require(pins.get(str(path))==sha(path),'own frozen396 canonical fast source')
        else:checked(path)
        values[name]=dict(path=str(path),sha256=sha(path))
    for name,module in tuple(sys.modules.items()):
        path=vars(module).get('__file__')if hasattr(module,'__dict__')else None
        if type(path)is str and Path(path).is_absolute()and Path(path).resolve().is_relative_to(here):
            path=Path(path).resolve()
            require(path.is_file()and pins.get(str(path))==sha(path),'all actual396 own loaded sources '+name)
    return values


def qualify_component(engine,native,stage,policy,output):
    from component396 import run
    return run(engine,native,stage,policy,output)
'''
    write('extension396.py', extension)
    component = '''"""Retain fresh173 math; select the own pinned helper origin and N24 lifecycle."""
import ast
from pathlib import Path
import component199 as original


def baseline(engine,native,stage,policy,output):
    source=Path(original.__file__).read_text()
    before='from extension199 import loaded_origins as assert_origins'
    after='from extension396 import loaded_origins as assert_origins'
    if source.count(before)!=1:raise RuntimeError('ROOT396 exact baseline origin anchor')
    projected=source.replace(before,after)
    if projected.replace(after,before)!=source:raise RuntimeError('ROOT396 baseline inverse')
    node=next(n for n in ast.parse(projected).body if isinstance(n,ast.FunctionDef)and n.name=='baseline')
    namespace=dict(vars(original))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
        __file__+'::original199-baseline-with-own-origin','exec'),namespace)
    return namespace['baseline'](engine,native,stage,policy,output)


def run(engine,native,stage,policy,output):
    proof=baseline(engine,native,stage,policy,output)
    if proof is not engine._v173_qualified_component:raise RuntimeError('ROOT396 fresh baseline identity')
    from binding396 import install
    binding=stage('ROOT396/current-target-backend-local-activation',lambda:install(engine,native))
    from actor396 import run as bounded
    extra=bounded(engine,native,stage,output)
    observed=engine._parent4_target391_binding.get('actual_metadata_matches',0)
    if not engine.is_draft and observed<=0:raise RuntimeError('ROOT396 current metadata checks missing')
    return dict(original173=proof,current_target_binding391=binding,actual_metadata_matches391=observed,
        bounded=extra,production_permission=False,full463=False,formal_Goodput=False,whole_AR=False)
'''
    write('component396.py', component)

    source = (B / 'tree-bounded-parent4-shared4-eager-diagnostic-v199/audit199.py').read_text()
    edits = [('from scope199 import cases,FAILURES,NATIVE_COUNTERS',
              'from scope396 import cases,FAILURES,NATIVE_COUNTERS', 1),
             ("len(bound['traces'])==15", "len(bound['traces'])==3*len(cases())", 1),
             ('exact5x3 traces', 'exact25x3 traces', 1),
             ("len(phase['owners'])<=16", "len(phase['owners'])<=24", 1),
             ('mixed16Q80', 'mixed24Q120', source.count('mixed16Q80')),
             ('def audit(output):', 'def base_audit(output):', 1)]
    audit = changed(source, edits)
    audit += '''

def audit(output):
    value=base_audit(output);output=Path(output)
    for rank in range(4):
        report=json.loads((output/f'rank{rank}-local-diagnostic.json').read_text())
        binding=report['component']['current_target_binding391']
        require(binding['actual_best376_target_backend']and not binding['best376_draft_claimed'],
            'own finite N24 uses independently bound M120 target')
        if rank:require(report['component']['actual_metadata_matches391']>0,'actual metadata witnessed')
        for path in (output/'bounded199').glob(f'rank{rank}-pass*.json.gz'):
            raw=json.loads(gzip.decompress(path.read_bytes()))
            for phase in raw['phases']:
                if rank:
                    shape=phase['target']['shape']
                    require(shape['M']==120 and shape['S']==24
                        and shape['family']=='static_FULL_M120_Q5_S24','actual current M120 finite phase')
    return dict(value,current_target_backend396='actual376 M120/Q5/S24 uniform KV',
        finite_N=24,finite_E=4,finite_R=list(range(25)),original173_fresh_baseline_retained=True,
        numerical_audit_reexecuted_by_HOST=False,production_permission=False,Raw=None,Goodput=None)
'''
    write('audit396.py', audit)
    projection['audit396.py'] = dict(
        source=str(B / 'tree-bounded-parent4-shared4-eager-diagnostic-v199/audit199.py'),
        edits=edits, original_all_proof_obligations_retained=True,
        added_checks='Actual391 target metadata and physical M120/S24 for every phase')

    source = (B / 'tree-parent4-current-M120-import-repair-v395/startup395.sh').read_text()
    edits = [('tree-parent4-current-M120-import-repair-v395/check395.py',
              P.name + '/check396.py', 1),
             ('tree-parent4-current-M120-import-repair-v395/runtime395.py',
              P.name + '/runtime396.py', 1),
             ('export PYTHONPATH=', 'export PYTHONPATH=' + str(P) + ':', 1)]
    write('startup396.sh', changed(source, edits))
    projection['startup396.sh'] = dict(
        source=str(B / 'tree-parent4-current-M120-import-repair-v395/startup395.sh'),
        edits=edits, inverse_verified=True)
    write('SOURCE_PROJECTION396.json', json.dumps(projection, indent=2) + '\n')
    print(json.dumps(dict(status='PREPARED_SOURCE_ONLY396', files=len(list(P.iterdir())),
                         N=24, E=4, R=list(range(25)), global_max_num_seqs=64,
                         NPU_used=False, production_permission=False)))


if __name__ == '__main__':
    main()
