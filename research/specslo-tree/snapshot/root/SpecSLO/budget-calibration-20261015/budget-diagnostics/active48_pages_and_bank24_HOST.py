"""Uninstalled 32/48 admission envelope and source-only24-owner proof bank.

These functions do not run an engine or grant larger-batch qualification.
The caller must supply only already-arrived requests in original priority order.
"""
import ast


def need(value, message):
    if not value:
        raise ValueError('active48 HOST candidate: '+message)


def charge(prompt, output, *, tail=9, block=128):
    need(all(type(v) is int for v in (prompt,output,tail,block))
         and prompt>0 and output>0 and tail>=0 and block>0, 'integer positive page inputs')
    return (prompt+output+tail+block-1)//block


def cohort(ready, prompt_lengths, output_limits, *, pool=544, scratch=64, token_limit=131072):
    ready = tuple(ready)
    need(len(set(ready)) == len(ready) and all(type(i) is int and 0<=i<len(prompt_lengths) for i in ready), 'unique ready-only owners')
    need(type(pool) is type(scratch) is type(token_limit) is int and 0<=scratch<pool and token_limit>0, 'explicit pool/scratch/token bounds')
    need(len(prompt_lengths)==len(output_limits), 'one output limit per prompt')
    costs = {i:charge(prompt_lengths[i],output_limits[i]) for i in ready}
    capacity = pool-scratch
    # Prefer a complete48 cohort; partial33..47 cohorts retain the existing32 cap.
    large = (len(ready)>=48 and sum(costs[i] for i in ready[:48])<=capacity
             and sum(prompt_lengths[i] for i in ready[:48])<=token_limit)
    limit = 48 if large else 32
    selected=[];pages=0;tokens=0
    for owner in ready[:limit]:
        if prompt_lengths[owner]>token_limit or costs[owner]>capacity:
            raise ValueError('one ready request exceeds declared capacity')
        if selected and (pages+costs[owner]>capacity or tokens+prompt_lengths[owner]>token_limit):
            break
        selected.append(owner)
        pages+=costs[owner]
        tokens+=prompt_lengths[owner]
    return dict(selected=tuple(selected), remaining=ready[len(selected):],
                declared_cap=limit, service_upper_pages=pages, scratch_upper_pages=scratch,
                total_upper_pages=pages+scratch, packed_prompt_tokens=tokens)


def bank24_source(original):
    tree=ast.parse(original)
    nodes=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef))
           and node.name in ('DualBank','build_bank')]
    need({node.name for node in nodes}=={'DualBank','build_bank'}, 'two original proof bank definitions')
    source='\n\n'.join(ast.get_source_segment(original,node) for node in nodes)+'\n'
    changes=(('range(32,64)','range(24,48)'),('range(32)','range(24)'),
             ('range(64)','range(48)'),('32+i','24+i'),('i+32','i+24'),
             ('owner%32','owner%24'),('range(463,475)','range(463,527)'),
             ('len(pages)==140','len(pages)==160'),('actual140/544','actual160/544'))
    adapted=source
    records=[]
    for before,after in changes:
        count=adapted.count(before)
        need(count>0,'original bank source anchor '+before)
        records.append(dict(before=before,after=after,count=count))
        adapted=adapted.replace(before,after)
    inverse=adapted
    for before,after in reversed(changes):
        inverse=inverse.replace(after,before)
    need(inverse==source,'complete byte inverse for both original bank bodies')
    ast.parse(adapted)
    return adapted,records
