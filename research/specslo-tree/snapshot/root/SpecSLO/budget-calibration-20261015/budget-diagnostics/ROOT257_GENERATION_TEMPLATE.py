"""LIVE target ordering: original strict whole-ticket priority by SLO class.

Original startup and all original component behavior are unchanged. Only the
new LIVE projection enables this policy, and its own counter reports one.
"""
import ast,copy,hashlib
from generation256 import patch as original_patch,descriptor as original_descriptor

def patch(source):
    base=original_patch(source);original=ast.parse(base);tree=copy.deepcopy(original)
    keywords=[];counters=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call)and ast.unparse(node.func)=='controller.build_plan':
            key=next((k for k in node.keywords if k.arg=='target_slo_first'),None)
            if key is None:raise RuntimeError('actual original SLO target keyword')
            keywords.append((key,copy.deepcopy(key.value)));key.value=ast.Constant(True)
        if isinstance(node,ast.Dict):
            for i,key in enumerate(node.keys):
                if isinstance(key,ast.Constant)and key.value=='spec_rhythm_target_slo_first_enabled':
                    counters.append((node,i,copy.deepcopy(node.values[i])));node.values[i]=ast.Constant(1)
    if len(keywords)!=1 or len(counters)!=1:raise RuntimeError('one actual LIVE target policy and truthful counter')
    value=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    for key,previous in keywords:key.value=previous
    for node,i,previous in counters:node.values[i]=previous
    if ast.dump(tree,include_attributes=False)!=ast.dump(original,include_attributes=False):
        raise RuntimeError('complete original LIVE AST inverse after two declared SLO changes')
    return value

def descriptor(source):
    result=original_descriptor(source);result.update(projected_source_sha256=hashlib.sha256(patch(source).encode()).hexdigest(),
        original_environment_target_SLO_first=False,actual_LIVE_target_SLO_first=True,
        truthful_actual_policy_counter=1,only_LIVE_policy_and_counter_changed=True,
        whole_ticket_budget_and_native_maths_unchanged=True)
    return result
