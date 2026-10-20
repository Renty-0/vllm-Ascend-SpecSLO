"""Original independent prefix/parent banks on actual full527 storage."""
import ast,copy
from base165 import B,checked,require
V149=B/'tree-repair-frontier-fusion-local-v149'
import sys
if str(V149) not in sys.path:sys.path.append(str(V149))
from scope_v149 import digest,Case
from reuse_v149 import contexts,kv_helpers,math_helpers

def owner_plan(case):
 return dict(candidate=list(case.candidate_ids),reference=list(case.reference_ids),
  actual_storage_rows=527,original_tree_scratch=list(range(463,527)),
  reserved_fixture_rows=list(case.candidate_ids+case.reference_ids),
  actual_pool=544,distinct_pages=4*case.count,prefix_tokens=128,
  teacher_forced_normal_suffix=True,natural_target_commit_claimed=False)

source=checked(V149/'bank_v149.py').read_text()
edits=(('e._cache_storage_sequence_capacity==128 and len(e.cache_allocation.block_tables)==128',
        'e._cache_storage_sequence_capacity==527 and len(e.cache_allocation.block_tables)==527'),
       ('tuple(e._tree_graph_scratch_sequence_ids)==()',
        'tuple(e._tree_graph_scratch_sequence_ids)==tuple(range(463,527))'),
       ('actual original V134 storage128/no implicit scratch reuse',
        'actual full527 storage, independently owned fixture and no implicit scratch reuse'))
for old,new in edits:
 require(source.count(old)==1,'exact full-storage independent bank anchor');source=source.replace(old,new)
nodes=[copy.deepcopy(n) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(V149/'bank_v149.py')+'::full527-P165','exec'),globals())
_exact_bank_delta=edits
