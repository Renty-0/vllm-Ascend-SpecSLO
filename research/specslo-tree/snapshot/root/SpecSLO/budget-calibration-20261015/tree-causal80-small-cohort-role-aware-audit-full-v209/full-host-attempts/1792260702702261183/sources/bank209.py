"""Keep real16/ref16; activate all64 original production scratch owners."""
import ast
from base209 import P173,functions,require
import bank173 as parent
from bank173 import _slots,_arrays,query_KV,prefix_evidence
globals().update({k:v for k,v in vars(parent).items()if k in('copy','tensor_hash','digest')})
# Generate the exact original173 bank first, then only expand its scratch lease.
text=parent.source
require(text.count('range(463,507)')==3 and text.count('len(pages)==108')==1,'exact real16 bank scratch/page anchors')
text=text.replace('range(463,507)','range(463,527)').replace('len(pages)==108','len(pages)==128').replace('actual108/544','actual128/544')
nodes=[n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in('DualBank','build_bank')]
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(P173/'bank173.py')+'::P209-all-production-scratch','exec'),globals())

def cases():
 from target_layout import _lengths
 from mapper173 import expected_geometry
 result=[]
 for n in range(9,16):
  full=parent.cases()[0];oldM,oldS,oldpad=expected_geometry(n,5*n,'causal')
  require((oldM,oldS)==(128,38),'original197 comparator remains128/38')
  result.append(dict(full,name='realN'+str(n),N=n,C=4*n,Q=5*n,budgets=[4]*n,
   M=80,S=16,dummy_lengths=_lengths(n,5*n,80,16),oldM=oldM,oldS=oldS,oldpad=oldpad,full_bank_case=full))
 return result
