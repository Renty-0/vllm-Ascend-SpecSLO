"""Two declared LIVE-generation limits; no constructor/config/clock mutation.

Effective verification candidates<=64 and target requests<=16 imply real
queries<=80. Original TargetAdmissionLimit128/32 and global draft192 remain
unchanged. Apply only after the full original live-source builder succeeds.
"""
import ast,copy,hashlib

def patch(source):
    original=ast.parse(source);tree=copy.deepcopy(original);keywords=[];inserted=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call)and ast.unparse(node.func)=='controller.build_plan':
            kw=next((x for x in node.keywords if x.arg=='max_target_requests'),None)
            if kw is None:raise RuntimeError('original controller target-row keyword required')
            keywords.append((kw,copy.deepcopy(kw.value)));kw.value=ast.Constant(value=16)
    class AddCap(ast.NodeTransformer):
        def visit_If(self,node):
            self.generic_visit(node)
            if ast.unparse(node.test)=='verification_roof == 0':
                statement=ast.parse('verification_roof = min(64, verification_roof)').body[0]
                inserted.append(statement);return[statement,node]
            return node
    AddCap().visit(tree)
    if len(keywords)!=1 or len(inserted)!=1:raise RuntimeError('exact one live row cap and one final verification-roof cap')
    result=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    for kw,old in keywords:kw.value=old
    def remove(node):
        for field,value in ast.iter_fields(node):
            if isinstance(value,list):
                kept=[x for x in value if not any(x is s for s in inserted)]
                setattr(node,field,kept)
                for x in kept:
                    if isinstance(x,ast.AST):remove(x)
            elif isinstance(value,ast.AST):remove(value)
    remove(tree)
    if ast.dump(tree,include_attributes=False)!=ast.dump(original,include_attributes=False):raise RuntimeError('entire original generation AST inverse')
    return result

def descriptor(source):
    result=patch(source)
    return dict(original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),canonical_source_sha256=hashlib.sha256(result.encode()).hexdigest(),
        effective_target_requests=16,effective_verification_candidates=64,real_query_bound=80,
        constructor_configuration_unchanged=True,original_admission_and_sampling_clocks_unchanged=True,global_selected_draft_limit=192,Native_qualification_flags_assigned=False)
