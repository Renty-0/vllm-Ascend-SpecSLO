"""Original checked255 tail layout with one longer alternate path.

Only HOST metadata is produced. New model math is not qualified.
"""
import ast,copy,inspect,textwrap
_LAYOUT=None

def prepared():
    global _LAYOUT
    if _LAYOUT is not None:return _LAYOUT
    import branch_tail285
    original=branch_tail285.layout
    node=ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    start=next(i for i,n in enumerate(node.body)if isinstance(n,ast.For)and ast.unparse(n.target)=='(plan, owner)')
    loop=node.body[start]
    begin=next(i for i,n in enumerate(loop.body)if isinstance(n,ast.If)and 'len(parents) != 5'in ast.unparse(n.test))
    stop=next(i for i,n in enumerate(loop.body)if isinstance(n,ast.Assign)and ast.unparse(n.targets[0])=='complete')
    old=copy.deepcopy(loop.body[begin:stop])
    replacement=ast.parse('''side=len(parents)-4
if not 1<=side<=4 or parents[:4]!=[-1,0,1,2] or parents[4] not in(-1,0,1,2) or parents[5:]!=list(range(4,len(parents)-1)):
    raise ValueError('HOST269 one primary4 plus one ancestor-closed alternate chain')
prefix=physical[0];first=parents[4]+2;last=first+side-1
if last>4 or physical!=list(range(prefix,prefix+q)) or logical!=list(range(prefix,prefix+5))+list(range(prefix+first,prefix+last+1)):
    raise ValueError('HOST269 exact original physical and RoPE ancestry')
groups.append(dict(owner=owner,original_rows=list(range(row,row+5)),tables=table,KV=prefix+5,masks=mask[:5]))
visible=(~mask[q-1]).nonzero().flatten().tolist()
if visible[:prefix]!=list(range(prefix)) or len(visible)!=logical[-1]+1 or visible[-side:]!=physical[5:]:
    raise ValueError('HOST269 complete ordered alternate-path visibility')
''').body
    loop.body[begin:stop]=replacement
    tail=next(n for n in loop.body if isinstance(n,ast.Assign)and ast.unparse(n.targets[0])=='canonical_mask')
    old_mask=copy.deepcopy(tail.value)
    tail.value=ast.parse('torch.arange(CONTEXT,device="cpu")[None,:]>=torch.tensor(logical[5:],device="cpu")[:,None]+1',mode='eval').body
    group=next(n for n in loop.body if isinstance(n,ast.Expr)and ast.unparse(n).startswith('groups.append')and 'row + 5'in ast.unparse(n)and 'canonical_mask'in ast.unparse(n))
    old_group=copy.deepcopy(group.value)
    group.value=ast.parse('groups.append(dict(owner=owner,original_rows=list(range(row+5,row+q)),tables=virtual,KV=logical[-1]+1,masks=canonical_mask))',mode='eval').body
    source=ast.unparse(ast.fix_missing_locations(node))+'\n'
    loop.body[begin:begin+len(replacement)]=old;tail.value=old_mask;group.value=old_group
    reference=ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    if ast.dump(node,include_attributes=False)!=ast.dump(reference,include_attributes=False):
        raise RuntimeError('HOST269 complete original255 AST inverse')
    ns=dict(original.__globals__);exec(compile(source,__file__+'::original255-with-alternate-chain','exec'),ns)
    _LAYOUT=ns['layout'];return _LAYOUT

def layout(*args,**kwargs):return prepared()(*args,**kwargs)
