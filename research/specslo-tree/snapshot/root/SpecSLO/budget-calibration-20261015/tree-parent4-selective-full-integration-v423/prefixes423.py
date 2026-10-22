"""Exact retained344 prefix fixtures; importing its B8 stack is unnecessary."""
import ast
import copy
import hashlib
from pathlib import Path

SOURCE = Path(__file__).parent.parent / 'tree-Q5-uniform-KV-bucket-branch-math-cost-v344/domain344.py'
SHA = '4f30db77956768382fd3844891e9ff1c9cabfece80a102c1342a24f3e86aad8d'


def load():
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA:
        raise RuntimeError('ROOT423 unchanged original344 prefix-fixture source')
    nodes = [copy.deepcopy(n) for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)
             and n.name in ('mid_prefixes', 'short_prefixes')]
    if len(nodes) != 2:
        raise RuntimeError('ROOT423 both complete original prefix fixtures required')
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(SOURCE) + '::unchanged-prefix-fixtures-only', 'exec'), namespace)
    return namespace['mid_prefixes'], namespace['short_prefixes']


mid_prefixes, short_prefixes = load()
