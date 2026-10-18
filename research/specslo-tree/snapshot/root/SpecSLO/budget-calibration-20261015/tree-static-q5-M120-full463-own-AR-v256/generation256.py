"""LIVE-onlyN24/C96 limits plus page-reserved priority-preserving admission."""
import ast,copy,hashlib


def patch(source):
    original=ast.parse(source);tree=copy.deepcopy(original);changes=[];inserted=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call)and ast.unparse(node.func)=='controller.build_plan':
            keyword=next((k for k in node.keywords if k.arg=='max_target_requests'),None)
            if keyword is None:raise RuntimeError('original actual target row cap')
            changes.append((keyword,copy.deepcopy(keyword.value)));keyword.value=ast.Constant(24)
    if len(changes)!=1:raise RuntimeError('one LIVE target row cap')
    admission=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='admit_available')
    loops=[n for n in admission.body if isinstance(n,ast.For)and ast.unparse(n.iter)=='admission_candidates']
    if len(loops)!=1:raise RuntimeError('one original priority admission loop')
    loop=loops[0];index=admission.body.index(loop)
    reserve=ast.parse('_root256_reserved_pages = 64').body[0]
    admission.body.insert(index,reserve);inserted.append((admission,reserve))
    additions=ast.parse('''
_root256_outputs = local_states[index].max_tokens
if type(_root256_outputs) is not int or _root256_outputs <= 0:
    raise RuntimeError('ROOT256 actual finite request output cap required')
_root256_pages = (int(local_states[index].prompt_length) + _root256_outputs + 5 + 127) // 128
if _root256_reserved_pages + _root256_pages > 544:
    counters['spec_rhythm_ROOT256_page_reserve_deferrals'] = counters.get('spec_rhythm_ROOT256_page_reserve_deferrals', 0) + 1
    break
_root256_reserved_pages += _root256_pages
''').body
    # Arrival eligibility remains the original first predicate. Reserve only
    # requests the same original loop is actually considering for admission.
    if not isinstance(loop.body[0],ast.If)or 'arrival_gated'not in ast.unparse(loop.body[0].test):
        raise RuntimeError('original first arrival eligibility predicate')
    loop.body[1:1]=additions
    for n in additions:inserted.append((loop,n))
    result=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    for keyword,value in changes:keyword.value=value
    for parent,node in reversed(inserted):parent.body.remove(node)
    if ast.dump(tree,include_attributes=False)!=ast.dump(original,include_attributes=False):
        raise RuntimeError('entire original LIVE body AST inverse after declared limits/reservation')
    return result


def descriptor(source):
    value=patch(source)
    return dict(original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        projected_source_sha256=hashlib.sha256(value.encode()).hexdigest(),
        effective_target_requests=24,effective_verification_candidates=96,original_target_query_limit=128,
        original_global_selected_draft_limit=192,active_cap=48,KV_pages=544,reserved_private_pages=64,
        request_order_and_priority_preserved=True,pending_requests_never_dropped=True,Native_flags_assigned=False)
