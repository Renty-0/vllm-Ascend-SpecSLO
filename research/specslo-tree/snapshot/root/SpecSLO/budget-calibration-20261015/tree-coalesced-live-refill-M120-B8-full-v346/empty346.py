"""One admission scheduling predicate; original clocks/publication remain exact."""
import ast,copy
from base209 import require

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
 groups=[v['sequence_ids']for v in trace['standalone_prefill_calls']]
 active=set();seen=set();part=[];group_index=0;cohorts=[];peak=0
 requests=trace['requests'];prompts=trace['tokenized_inputs']
 def owner_pages(index):
  outputs=requests[index]['max_tokens']
  require(type(outputs)is int and outputs>0,'actual finite original request output limit')
  return (len(prompts[index])+outputs+8+127)//128
 for event in trace['stream_events']:
  i=event['request_index']
  require(type(i)is int and 0<=i<trace['scope_count'],'actual original owner index')
  if i not in seen:
   if not part:
    require(group_index<len(groups),'actual root group belongs to original eager helper')
    owners=list(groups[group_index])
    require(0<len(owners)<=48 and len(owners)==len(set(owners))
     and not(set(owners)&seen),'once-only fresh prefill owner group')
    require(len(active)+len(owners)<=48,'actual48 active limit while incumbents survive')
    reserved=64+sum(owner_pages(j)for j in active|set(owners))
    require(reserved<=544,'incumbent and all new owner maximum pages fit actual544')
    cohorts.append(dict(ordinal=group_index,owners=owners,incumbent_count=len(active),
     incumbent_owners=sorted(active),conservative_reserved_pages=reserved))
   require(i==groups[group_index][len(part)],'original helper order and immediate once-root publication')
   if 'first_token_wall_seconds160'in event:
    require(event['first_token_wall_seconds160']>=trace['absolute_arrivals'][i],
     'no root published before original absolute arrival')
   seen.add(i);active.add(i);part.append(i);peak=max(peak,len(active))
   if len(part)==len(groups[group_index]):part=[];group_index+=1
  else:require(not part,'no incumbent callback interleaved into original root group')
  if event['finished']:
   require(i in active,'finished owner was genuinely active');active.remove(i)
 require(not active and not part and group_index==len(groups)and seen==set(range(trace['scope_count'])),
  'all original finite requests drained without dropping an infeasible owner')
 return dict(cohorts=cohorts,cohort_count=len(cohorts),all_arrivals_retained=True,
  no_active_refill=not any(row['incumbent_count']for row in cohorts),
  actual_inflight_refill_groups=sum(row['incumbent_count']>0 for row in cohorts),
  actual_peak=peak,source_clocks_unchanged=True,minimum_refill_policy=4,
  no_extra_root_hold_or_callback=True,no_per_prefill_duration_inference=True,
  actual_incumbent_and_new_pages_checked=True,actual_arrival_publication_checked=True)
