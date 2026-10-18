"""One admission scheduling predicate; original clocks/publication remain exact."""
import ast,copy
from base205 import require

def admission(tree):
 rows=[n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='admit_available']
 require(len(rows)==1,'one original admission function');return rows[0]
def patch(source):
 tree=ast.parse(source);fn=admission(tree)
 require(isinstance(fn.body[0],ast.Assign)and ast.unparse(fn.body[0].targets[0])=='closed'
  and 'receive_live_admissions' in ast.unparse(fn.body[0].value),'original receive before admission predicate')
 added=ast.parse('if active:\n return closed').body[0]
 require(not any(ast.dump(n,include_attributes=False)==ast.dump(added,include_attributes=False)for n in fn.body),'not patched twice')
 fn.body.insert(1,added);result=ast.unparse(ast.fix_missing_locations(tree))+'\n'
 assert_inverse(source,result);return result
def unpatch(source):
 tree=ast.parse(source);fn=admission(tree);want=ast.parse('if active:\n return closed').body[0]
 require(len(fn.body)>1 and ast.dump(fn.body[1],include_attributes=False)==ast.dump(want,include_attributes=False),'one exact common active guard after receive')
 del fn.body[1];return ast.unparse(ast.fix_missing_locations(tree))+'\n'
def assert_inverse(original,adapted):
 require(ast.dump(ast.parse(original),include_attributes=False)==ast.dump(ast.parse(unpatch(adapted)),include_attributes=False),'whole original decode AST inverse; no clock/publication/body edit');return True

def audit_trace(trace):
 groups=[v['sequence_ids']for v in trace['standalone_prefill_calls']];active=set();seen=set();part=[];group_index=0;cohorts=[]
 for event in trace['stream_events']:
  i=event['request_index']
  if i not in seen:
   if not part:
    require(not active,'ordinary wholeprompt prefill never interrupts an incumbent');require(group_index<len(groups),'actual root group belongs to helper')
    cohorts.append(dict(ordinal=group_index,owners=list(groups[group_index]),incumbent_count=0))
   require(i==groups[group_index][len(part)],'original helper owner order and immediate once-root publication')
   seen.add(i);active.add(i);part.append(i)
   if len(part)==len(groups[group_index]):part=[];group_index+=1
  else:require(not part,'no incumbent callback can interleave a root group')
  if event['finished']:
   require(i in active,'finished owner was genuinely active');active.remove(i)
 require(not active and not part and group_index==len(groups)and seen==set(range(trace['scope_count'])),'all original finite requests drained without drop')
 require(all(0<len(row['owners'])<=32 for row in cohorts),'real service capacity32, no minimum batch size promise')
 return dict(cohorts=cohorts,cohort_count=len(cohorts),all_arrivals_retained=True,no_active_refill=True,source_clocks_unchanged=True,
  no_extra_root_hold_or_callback=True,no_per_prefill_duration_inference=True)
