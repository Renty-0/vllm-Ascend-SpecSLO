"""Retain original335 owned synthetic lifecycle; change only source/scope/audit."""
import ast
import copy
from pathlib import Path
import re


B = Path('/root/SpecSLO/budget-calibration-20261015')
OLD = B / 'tree-FIA-Q4-group-and-query-isolation-v335'
NEW = B / 'tree-FIA-multi-owner-page-offset-invariance-v363'


def checked(source, changes):
    projected = source
    for before, after, count in changes:
        if projected.count(before) != count:
            raise RuntimeError('exact363 source anchor required: ' + before)
        projected = projected.replace(before, after)
    inverse = projected
    for before, after, _ in reversed(changes):
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise RuntimeError('complete original335 source inverse required')
    return projected


def main():
    assert not (NEW / 'FREEZE363.json').exists() and not (NEW / 'guard363.py').exists()
    source = (OLD / 'guard335.py').read_text()
    projected = checked(source, (
        ('from probe335 import configurations', 'from probe363 import configurations', 1),
        ("HERE/'FREEZE335.json'", "HERE/'FREEZE363.json'", 3),
        ("HERE/'SOURCE335.tar.gz'", "HERE/'SOURCE363.tar.gz'", 2),
        ("HERE/'startup335.sh'", "HERE/'startup363.sh'", 1),
        ('MAX_SECONDS = 300', 'MAX_SECONDS = 600', 1),
        ('MAX_TAG_BYTES = 16*1024**2', 'MAX_TAG_BYTES = 64*1024**2', 1),
        ('MAX_AGGREGATE_BYTES = 128*1024**2', 'MAX_AGGREGATE_BYTES = 256*1024**2', 1),
        ('synthetic-FIA-Q4-Q5-group-query-invariance', 'synthetic-FIA-Q5-S24-multi-owner-page-offset-invariance', 1),
        ('synthetic_Q4_and_Q5=True, groups=[24, 32, 33], rows=864',
         'synthetic_multi_owner_page_offsets=True, groups=[24], KV_heads=[2, 3], rows=1920', 1),
        ('ROOT.V335.synthetic-FIA-supervisor.v1', 'ROOT.V363.synthetic-FIA-page-offset-supervisor.v1', 1),
        ('ROOT335 supervisor received', 'ROOT363 supervisor received', 1),
        ('synthetic Q4/Q5 group/query invariance; no models, service requests or performance',
         'synthetic Q5/S24 multi-owner page offsets with original FULL2560 controls; no model or serving metric', 1),
        ('bounded300second', 'bounded600second', 1),
        ('within300seconds', 'within600seconds', 1),
        ("    own = json.loads((HERE/'FREEZE363.json').read_text())",
         "    own = json.loads((HERE/'FREEZE363.json').read_text())\n"
         "    host = json.loads((HERE/'HOST363_RESULT.json').read_text())\n"
         "    require(host['status'].startswith('PASS_HOST') and host['NPU_used'] is False\n"
         "            and host['probe_sha256'] == sha(HERE/'probe363.py'), 'exact363 HOST metadata/source prerequisite')", 1)))
    tree = ast.parse(projected)
    before = copy.deepcopy(tree)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'actual_audit')
    previous = function.body
    function.body = ast.parse('from audit363 import audit\nreturn audit(directory)\n').body
    changed = copy.deepcopy(tree)
    function.body = previous
    if ast.dump(tree, include_attributes=False) != ast.dump(before, include_attributes=False):
        raise RuntimeError('complete original335 audit-function AST inverse required')
    guard = ast.unparse(ast.fix_missing_locations(changed)) + '\n'
    # The original operator's source hash literals live only in the replaced
    # old auditor; all unchanged lifecycle declarations retain their code.
    startup = checked((OLD / 'startup335.sh').read_text(), (
        (str(OLD), str(NEW), 1), ('probe335.py', 'probe363.py', 1)))
    (NEW / 'guard363.py').write_text(guard)
    (NEW / 'startup363.sh').write_text(startup)
    print('additive363 original owned guard/startup constructed; no device launch')


if __name__ == '__main__':
    main()
