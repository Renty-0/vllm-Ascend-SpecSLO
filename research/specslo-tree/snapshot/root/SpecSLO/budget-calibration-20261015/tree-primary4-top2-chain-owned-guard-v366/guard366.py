"""Additive HOST-only adapter for byte-identical inherited dependency pins.

ROOT365's component code is unchanged. Its first source-only construction
rejected three dependencies because they are already in parent333. Require
their pins to be identical, then retain the complete union. No numerical or
Native qualification body is projected here.
"""
import argparse
import ast
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

HERE = Path(__file__).resolve().parent
COMPONENT = HERE.parent / 'tree-primary4-top2-chain-device-v365'
sys.path.insert(0, str(COMPONENT))
import guard365 as component

EXPECTED_FREEZE = None
EXPECTED_ARCHIVE = None
sha = component.sha
require = component.require


def project_prepared():
    source = (COMPONENT / 'guard365.py').read_text()
    node = copy.deepcopy(next(n for n in ast.parse(source).body
                              if isinstance(n, ast.FunctionDef) and n.name == 'prepared'))
    original = copy.deepcopy(node)
    matches = [n for n in ast.walk(node) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Name) and n.func.id == 'require'
               and len(n.args) == 2 and isinstance(n.args[1], ast.Constant)
               and n.args[1].value == 'disjoint immutable additive files']
    require(len(matches) == 1, 'exact original source-only overlap declaration')
    call = matches[0]
    old_test = copy.deepcopy(call.args[0])
    call.args[0] = ast.parse('all(sources[path] == own_sources[path] for path in set(sources) & set(own_sources))', mode='eval').body
    projected = copy.deepcopy(node)
    call.args[0] = old_test
    require(ast.dump(node, include_attributes=False) == ast.dump(original, include_attributes=False),
            'entire original prepared AST inverse')
    namespace = dict(vars(component))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[projected], type_ignores=[])),
                 __file__ + '::identical-dependency-pin-union', 'exec'), namespace)
    return namespace['prepared']


def prepared():
    require(sha(HERE / 'FREEZE366.json') == EXPECTED_FREEZE, 'exact366 source freeze')
    require(sha(HERE / 'SOURCE366.tar.gz') == EXPECTED_ARCHIVE, 'exact366 source archive')
    own = json.loads((HERE / 'FREEZE366.json').read_text())
    for filename, expected in own['files'].items():
        require(sha(filename) == expected, 'immutable366 source ' + filename)
    inventory = dict(own['files'], **{str(HERE / 'FREEZE366.json'): EXPECTED_FREEZE})
    with tarfile.open(HERE / 'SOURCE366.tar.gz', 'r:gz') as saved:
        members = saved.getmembers()
        require({m.name for m in members} == {p.lstrip('/') for p in inventory}
                and len(members) == len(inventory), 'complete366 archive inventory')
        for member in members:
            require(member.isfile() and hashlib.sha256(saved.extractfile(member).read()).hexdigest()
                    == inventory['/' + member.name], 'exact366 archived bytes')
    pins = json.loads((COMPONENT / 'SOURCE365_RECEIPT.json').read_text())
    component.EXPECTED_FREEZE = pins['freeze_sha256']
    component.EXPECTED_ARCHIVE = pins['archive_sha256']
    original, sources, readiness = project_prepared()()
    for filename in set(sources) & set(inventory):
        require(sources[filename] == inventory[filename], 'identical inherited366 dependency')
    sources.update(inventory)
    readiness.update(guard366_freeze_sha256=EXPECTED_FREEZE,
                     guard366_archive_sha256=EXPECTED_ARCHIVE,
                     original365_neural_files_unchanged=True,
                     dependency_union_requires_identical_SHA256=True,
                     source_members=len(sources))
    return original, sources, readiness


def main():
    global EXPECTED_FREEZE, EXPECTED_ARCHIVE
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--expected-freeze-sha256', required=True)
    parser.add_argument('--expected-archive-sha256', required=True)
    args = parser.parse_args()
    EXPECTED_FREEZE, EXPECTED_ARCHIVE = args.expected_freeze_sha256, args.expected_archive_sha256
    require(re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', args.tag) is not None
            and all(re.fullmatch('[0-9a-f]{64}', x) for x in (EXPECTED_FREEZE, EXPECTED_ARCHIVE)),
            'safe explicit tag and source pins')
    prepared()
    namespace = component.execution_namespace()
    namespace['prepared'] = prepared
    lock = component.LOCK.open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        namespace['execute_locked'](args.tag)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
