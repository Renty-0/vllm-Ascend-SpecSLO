"""Independent real16/reference16 owner banks with disjoint old-layout scratch."""
import copy,ast
from base173 import BASE,checked,require
from scope_v146 import digest
from dual_bank_v146 import _arrays,_slots,slot_values,tensor_hash,prefix_evidence,query_KV

source=checked(BASE/'dual_bank_v146.py').read_text()
edits=(('32+i','16+i'),('range(463,475)','range(463,507)'),('range(64)','range(32)'),
 ('owner%32','owner%16'),('i+32','i+16'),('range(32,64)','range(16,32)'),('range(32)','range(16)'),
 ('len(pages)==140','len(pages)==108'),('actual140/544','actual108/544'))
# Explicit order avoids recursively reducing a newly rewritten range(32).
source=checked(BASE/'dual_bank_v146.py').read_text()
source=source.replace('range(32,64)','range(16,32)').replace('range(32)','range(16)')
source=source.replace('range(64)','range(32)').replace('32+i','16+i').replace('i+32','i+16').replace('owner%32','owner%16').replace('i%32','i%16')
source=source.replace('range(463,475)','range(463,507)').replace('len(pages)==140','len(pages)==108').replace('actual140/544','actual108/544')
nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('DualBank','build_bank')]
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(BASE/'dual_bank_v146.py')+'::P173-real16','exec'),globals())

def cases():
 from scope_v146 import cases as original
 owners=copy.deepcopy(original()[0]['owners'][:16])
 for i,row in enumerate(owners):row['owner']=i;row['reference_owner']=i+16
 common=dict(index=0,N=16,C=64,Q=80,budgets=[4]*16,owners=owners,
  initial_root_teacher_forced_not_own_AR=True,subsequent_draft_and_verdict_actual=True)
 return [dict(common,name='new80',M=80,S=16,dummy_lengths=[]),
         dict(common,name='old128',M=128,S=38,dummy_lengths=[3]*4+[2]*18)]
