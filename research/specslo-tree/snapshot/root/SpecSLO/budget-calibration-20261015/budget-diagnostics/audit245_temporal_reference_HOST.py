"""Correct the time index of one full page-table check; retain every AR gate.

NativePrefixCache allocates prompt pages first and adds each queried logical
block on demand. The retained cohort table is saved after all255 steps. An
earlier sampled table therefore has zero-page mappings for future pages that
exist in the final table. Checking those future IDs at step0 was incorrect.
"""
import ast,hashlib,importlib.util
from pathlib import Path

ORIGINAL=Path('/root/SpecSLO/budget-calibration-20261015/tree-static-q5-full463-own-AR-v245/audit_reference245.py')
ORIGINAL_SHA='a011c3fc58f531c36f6d96c15670c325b57b391cd8e6f698ff7b096b6e9b74e2'


def expected_allocated_table(final_table,position,zero):
    if len(final_table)!=32 or type(position)is not int or not 0<=position<2560:
        raise RuntimeError('complete original own-reference page-table dimensions')
    used=(position+5+127)//128
    if not all(type(v)is int and 0<=v<544 for v in final_table[:used]):
        raise RuntimeError('all pages needed by five actually written queries exist')
    return final_table[:used]+[zero]*(32-used)


def prepared():
    source=ORIGINAL.read_text()
    if hashlib.sha256(source.encode()).hexdigest()!=ORIGINAL_SHA:
        raise RuntimeError('unchanged failed original full245 reference auditor')
    before="meta['block_tables'][index]==[v if v>=0 else zero for v in tables[owner]]"
    after="meta['block_tables'][index]==expected_allocated_table(tables[owner],positions[5*index],zero)"
    if source.count(before)!=1:raise RuntimeError('one exact temporal full-table predicate')
    changed=source.replace(before,after)
    if changed.replace(after,before)!=source:raise RuntimeError('entire original reference-audit source inverse')
    namespace=dict(expected_allocated_table=expected_allocated_table)
    exec(compile(changed,str(Path(__file__).resolve())+'::all-original-gates-with-temporal-table','exec'),namespace)
    return namespace['audit']


audit=prepared()
