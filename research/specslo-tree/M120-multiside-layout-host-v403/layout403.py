"""HOST-only multi-side-leaf layout; original M120 and1792 copy slots.

Importing this module installs nothing. New model/KV qualification, live
resource packing and complete own-AR are required before any runtime use.
"""
import hashlib
import inspect
from pathlib import Path
import textwrap

import branch_tail255 as original


def sources():
    source = textwrap.dedent(inspect.getsource(original.layout))
    begin = source.index('        if len(parents)!=5 or')
    last = "        groups.append(dict(owner=owner,original_rows=[row+5],tables=virtual,KV=logical[5]+1,masks=canonical_mask));row+=q\n"
    end = source.index(last, begin) + len(last)
    previous = source[begin:end]
    replacement = '''        if (not 5<=len(parents)<=8 or parents[:4]!=[-1,0,1,2]
            or any(parent not in(-1,0,1,2) for parent in parents[4:])
            or len(set(parents[4:]))!=len(parents)-4):
            raise ValueError('qualified four-node spine plus distinct alternate leaves only')
        prefix=physical[0]
        expected_logical=list(range(prefix,prefix+5))+[prefix+parent+2 for parent in parents[4:]]
        if physical!=list(range(prefix,prefix+q)) or logical!=expected_logical:
            raise ValueError('original complete primary/side physical and RoPE layout')
        groups.append(dict(owner=owner,original_rows=list(range(row,row+5)),tables=table,KV=prefix+5,masks=mask[:5]))
        for query in range(5,q):
            visible=(~mask[query]).nonzero().flatten().tolist()
            if visible[:prefix]!=list(range(prefix)) or len(visible)!=logical[query]+1 or visible[-1]!=physical[query]:
                raise ValueError('exact alternate leaf visibility and ancestor order')
            complete=prefix//BLOCK;tail=visible[complete*BLOCK:];n=(len(tail)+BLOCK-1)//BLOCK
            own=pages[consumed:consumed+n]
            if len(own)!=n:raise ValueError('private branch-tail page bound')
            consumed+=n;virtual=list(table)
            if complete+n>len(virtual):raise ValueError('canonical logical KV exceeds model context')
            for j,page in enumerate(own):virtual[complete+j]=page
            for j,pos in enumerate(tail):
                physical_page=table[pos//BLOCK]
                if not 0<=physical_page<544:raise ValueError('branch reads only original allocated KV')
                source.append(physical_page*BLOCK+pos%BLOCK);dest.append(own[j//BLOCK]*BLOCK+j%BLOCK)
            canonical_mask=torch.arange(CONTEXT,device='cpu')[None,:]>=logical[query]+1
            groups.append(dict(owner=owner,original_rows=[row+query],tables=virtual,KV=logical[query]+1,masks=canonical_mask))
        row+=q
'''
    after = source[:begin] + replacement + source[end:]
    anchor = '    branches=sum(is_fork(p)for p in plans)\n'
    guard = '''    if type(copy_slots)is not int or copy_slots!=1792:
        raise ValueError('unchanged fixed1792 copy capacity required')
    real=[p for p,owner in zip(plans,ids,strict=True) if owner not in scratch_ids]
    if (len(real)>24 or sum(p.candidate_budget for p in real)>96
        or sum(1+max(0,p.candidate_budget-4) for p in real)>24):
        raise ValueError('complete M120/S24/N24/C96 real primary/side resources required')
    if any(p.candidate_budget>4 and not is_fork(p) for p in real):
        raise ValueError('real linear spines at most4; deeper budgets require qualified alternate leaves')
'''
    if after.count(anchor) != 1:
        raise RuntimeError('one original whole-layout resource boundary')
    after = after.replace(anchor, guard + anchor)
    if after.replace(guard, '').replace(replacement, previous) != source:
        raise RuntimeError('complete original one-side layout inverse')
    return source, after


def construct():
    _, source = sources()
    namespace = dict(vars(original))
    exec(compile(source, str(Path(__file__).resolve()) + '::multiside-layout-only',
                 'exec', dont_inherit=True), namespace)
    return namespace['layout']


layout = construct()


def descriptor():
    before, after = sources()
    return dict(original_sha256=hashlib.sha256(before.encode()).hexdigest(),
                projected_sha256=hashlib.sha256(after.encode()).hexdigest(),
                M=120, physical_target_segments=24, fixed_copy_slots=1792,
                spine_length=4, maximum_alternate_leaves_per_owner=4,
                conservative_global_side_leaf_limit=13,
                exact_group24_and_candidate96_and_copy1792_bounds=True,
                original_layout_function_modified=False,
                NPU_qualified=False, production_permission=False, Raw=None, Goodput=None)
