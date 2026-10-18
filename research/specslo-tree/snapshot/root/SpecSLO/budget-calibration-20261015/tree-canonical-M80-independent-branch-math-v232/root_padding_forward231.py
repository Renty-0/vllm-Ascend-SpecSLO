"""Private host-forward projection: allow only typed physical root padding.

The original positive-budget conditions stay unchanged for all real rows.
No frozen source file is edited. The AST inverse checks the three narrow
sites needed to validate and retain a typed physical padding row.
"""
import ast,copy,inspect,textwrap

def project(source):
    original=ast.parse(textwrap.dedent(source));tree=copy.deepcopy(original);changed=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.If)and ast.unparse(node.test)=='not 1 <= active <= expected or len(row) not in (active, expected)':
            old=copy.deepcopy(node.test)
            node.test=ast.parse('(not (1 <= active <= expected or _root_only_padding_allowed231(plan, row_id=row_id, real_count=real_count))) or len(row) not in (active, expected)',mode='eval').body
            changed.append((node,'test',old))
        if isinstance(node,ast.Assign)and ast.unparse(node)=='plan = pack_selected_tree_plan(plan, range(active))':
            old=copy.deepcopy(node.value)
            node.value=ast.parse('plan if _root_only_padding_allowed231(plan, row_id=row_id, real_count=real_count) else pack_selected_tree_plan(plan, range(active))',mode='eval').body
            changed.append((node,'value',old))
        if isinstance(node,ast.If)and ast.unparse(node.test)=='not 1 <= active_count <= node_count':
            old=copy.deepcopy(node.test)
            node.test=ast.parse('not (1 <= active_count <= node_count or _root_only_padding_allowed231(plan, row_id=row_id, real_count=real_count))',mode='eval').body
            changed.append((node,'test',old))
    if len(changed)!=3:raise RuntimeError('exact three original input/packing/output anchors required')
    projected=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    for node,attribute,old in changed:setattr(node,attribute,old)
    if ast.dump(tree,include_attributes=False)!=ast.dump(original,include_attributes=False):raise RuntimeError('whole original target-forward AST inverse')
    return projected

def host_installer(original_installer):
    """Use only in the entry installer, before its remaining target wrappers."""
    def install(native):
        from padding231 import allowed_dummy
        original_method=native.NativePearlEngine.target_tree_forward
        source=textwrap.dedent(inspect.getsource(original_method))
        before=('self.model.make_tree_attention_metadata(\n'
            '        plan_list,\n        roots,\n        padded_candidates,\n'
            '        self.cache_allocation.block_tables,')
        after=('_dynamic_tree_make_host_metadata(\n        self,\n'
            '        plan_list,\n        roots,\n        padded_candidates,')
        if source.count(before)!=1:raise RuntimeError('exact unchanged original host-metadata substitution')
        source=source.replace(before,after)
        original_installer(native) # retain its exact counted builder and original setup.
        namespace=vars(native);namespace['_root_only_padding_allowed231']=allowed_dummy
        exec(compile(project(source),__file__+'::host-and-root-padding','exec'),namespace)
        native.NativePearlEngine.target_tree_forward=namespace['target_tree_forward']
    return install
