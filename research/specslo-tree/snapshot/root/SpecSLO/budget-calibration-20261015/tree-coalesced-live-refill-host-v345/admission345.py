"""Bounded live refill at the original admission boundary, with incumbent pages."""
import ast
import copy
import hashlib


def patch(source, minimum=4):
    if type(minimum) is not int or not 1 <= minimum <= 8:
        raise ValueError('declared live refill group in 1..8')
    original = ast.parse(source)
    tree = copy.deepcopy(original)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'admit_available']
    if len(functions) != 1:
        raise RuntimeError('one original actual admission function')
    fn = functions[0]
    guard = ast.parse('if active:\n return closed').body[0]
    if ast.dump(fn.body[1]) != ast.dump(guard):
        raise RuntimeError('exact previous empty-active-only guard required')
    removed = fn.body.pop(1)
    changes = []
    inserted = []

    def replace(parent, old, replacements):
        index = parent.body.index(old)
        parent.body[index:index+1] = replacements
        changes.append((parent, index, old, len(replacements)))

    reserves = [n for n in fn.body if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == '_root256_reserved_pages']
    if len(reserves) != 1 or ast.unparse(reserves[0].value) != '64':
        raise RuntimeError('original private64 page reservation')
    replacement = ast.parse('''
def _root345_pages_for_owner(owner):
    outputs = local_states[owner].max_tokens
    prompt = local_states[owner].prompt_length
    if type(outputs) is not int or outputs <= 0 or type(prompt) is not int or prompt <= 0:
        raise RuntimeError('ROOT345 finite actual owner prompt and output cap required')
    return (prompt + outputs + 8 + 127) // 128
_root345_reserved_owners = set(active) | (set(prefetched) & set(pending_admission))
_root256_reserved_pages = 64 + sum(_root345_pages_for_owner(owner) for owner in _root345_reserved_owners)
if _root256_reserved_pages > 544:
    raise RuntimeError('ROOT345 incumbent reservation exceeds actual544 pages')
_root345_page_limited = False
''').body
    replace(fn, reserves[0], replacement)
    loop = next(n for n in fn.body if isinstance(n, ast.For) and ast.unparse(n.iter) == 'admission_candidates')
    page_assignment = [n for n in loop.body if isinstance(n, ast.Assign)
                       and ast.unparse(n.targets[0]) == '_root256_pages']
    if len(page_assignment) != 1 or not any(
            f'+ {tail} +' in ast.unparse(page_assignment[0].value) for tail in (5, 8)):
        raise RuntimeError('original per-new-owner future5 or future8 reservation')
    replace(loop, page_assignment[0], ast.parse(
        '_root256_pages = 0 if index in _root345_reserved_owners else _root345_pages_for_owner(index)').body)
    page_guard = next(n for n in loop.body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == '_root256_reserved_pages + _root256_pages > 544')
    mark = ast.parse('_root345_page_limited = True').body[0]
    page_guard.body.insert(0, mark)
    inserted.append((page_guard, mark))
    coalescer = next(n for n in fn.body if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'admitted and prefill_coalescing')
    gate = ast.parse(f'''
if active and admitted and len(admitted) < min({minimum}, len(pending_admission)) and not token_budget_saturated and not _root345_page_limited:
    counters['spec_rhythm_ROOT345_coalesced_refill_deferrals'] = counters.get('spec_rhythm_ROOT345_coalesced_refill_deferrals', 0) + 1
    admitted = []
''').body[0]
    fn.body.insert(fn.body.index(coalescer), gate)
    inserted.append((fn, gate))
    result = ast.unparse(ast.fix_missing_locations(tree))+'\n'
    for parent, node in reversed(inserted):
        parent.body.remove(node)
    for parent, index, old, count in reversed(changes):
        parent.body[index:index+count] = [old]
    fn.body.insert(1, removed)
    if ast.dump(tree, include_attributes=False) != ast.dump(original, include_attributes=False):
        raise RuntimeError('whole LIVE source inverse after declared refill/page changes')
    return result


def descriptor(source, minimum=4):
    result = patch(source, minimum)
    return dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),
                projected_sha256=hashlib.sha256(result.encode()).hexdigest(),
                minimum_refill=minimum, active_limit=48, actual_pages=544, private_pages=64,
                reservation_includes_incumbents=True, reservation_includes_prefetched_pending=True,
                future_tail=8, original_arrival_offsets_and_clock_expressions=True,
                original_incumbent_prefill_time_accounting=True,
                original_prompt_prefill_helper=True, original_priority_order=True,
                model_qualified=False, whole463_qualified=False, Goodput=None)
