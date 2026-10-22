"""All original independent AR checks at actual per-position physical M."""
import ast,copy,hashlib,inspect
import audit_reference425 as original
from geometry_history460 import build

def expected_mask(positions):
    original.need(len(positions)in(80,120),'complete selected reference position list')
    groups=len(positions)//5;mask=bytearray([1])*(groups*10*4096)
    for group in range(groups):
        count=positions[5*group]+1
        original.need(0<count<=2560,'every own first query fits original static extent')
        start=group*10*4096;mask[start:start+count]=bytearray(count)
        for query in range(1,5):mask[start+query*4096]=0
    return hashlib.sha256(mask).hexdigest()

def constructed(*,adaptive_required=True):
    if type(adaptive_required)is not bool:raise TypeError('ROOT460 explicit geometry audit mode')
    before=ast.parse(inspect.getsource(original.audit));tree=copy.deepcopy(before)
    fn=tree.body[0];leader_assign=next(n for n in fn.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='leader'for t in n.targets))
    at=fn.body.index(leader_assign)+1
    geometry=ast.parse('geometry460=build_geometry460(leader,adaptive_required=ADAPTIVE_REQUIRED460)').body[0]
    fn.body.insert(at,geometry)
    loops=[n for n in ast.walk(tree)if isinstance(n,ast.For)and ast.unparse(n.target)=='begin'and ast.unparse(n.iter)=='range(0, len(owners), 16)']
    if len(loops)!=1:raise RuntimeError('ROOT460 one original independent reference history loop')
    loop=loops[0];saved_loop=copy.deepcopy(loop)
    if not isinstance(loop.body[0],ast.Assign)or ast.unparse(loop.body[0].targets[0])!='selected':
        raise RuntimeError('ROOT460 exact original owner chunk selection')
    loop.body[0].targets[0].id='chunk460'
    step_loop=loop.body[1]
    if not isinstance(step_loop,ast.For)or ast.unparse(step_loop.target)!='step':
        raise RuntimeError('ROOT460 original255 independent prediction steps')
    counts={24:0,120:0,121:0,'backend':0}
    class Dimensions(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value)is int and node.value in(24,120,121):
                counts[node.value]+=1
                expression={24:'groups460',120:'physical_M',121:'physical_M+1'}[node.value]
                return ast.copy_location(ast.parse(expression,mode='eval').body,node)
            if node.value=='static_FULL_q5_M120_S24_KV2560':
                counts['backend']+=1
                if adaptive_required:
                    return ast.copy_location(ast.Constant('independent_static_FULL_q5_actual_M80_or_M120_KV2560'),node)
                return node
            return node
    body=[Dimensions().visit(copy.deepcopy(node))for node in step_loop.body]
    if counts[120]!=1 or counts[121]!=1 or counts[24]<3 or counts['backend']!=1:
        raise RuntimeError('ROOT460 exact original metadata-dimension predicates '+repr(counts))
    extra=ast.parse("need((not ADAPTIVE_REQUIRED460) or (saved['physical_M']==physical_M and saved['real_owners']==selected),'actual saved physical geometry and current owner subset')").body[0]
    # saved assignment, cursor increment, heads assignment precede old checks.
    heads=next(i for i,node in enumerate(body)if isinstance(node,ast.Assign)and ast.unparse(node.targets[0])=='heads')
    body.insert(heads+1,extra)
    prefix=ast.parse("selected=[i for i in chunk460 if geometry460['schedules'][owners[i]][step+1]==physical_M]\nif not selected:continue\ngroups460=physical_M//5").body
    physical_loop=ast.For(target=ast.Name(id='physical_M',ctx=ast.Store()),iter=ast.Tuple(elts=[ast.Constant(80),ast.Constant(120)],ctx=ast.Load()),body=prefix+body,orelse=[])
    step_loop.body=[physical_loop]
    projected=copy.deepcopy(tree)
    loop.__dict__.update(copy.deepcopy(saved_loop.__dict__))
    fn.body.remove(geometry)
    if ast.dump(tree,include_attributes=False)!=ast.dump(before,include_attributes=False):
        raise RuntimeError('ROOT460 complete original retained-AR-auditor inverse')
    ns=dict(vars(original),expected_static_mask_sha256=expected_mask,build_geometry460=build,ADAPTIVE_REQUIRED460=adaptive_required)
    exec(compile(ast.fix_missing_locations(projected),__file__+'::original-all-AR-checks-actual-geometry','exec'),ns)
    return ns['audit']

def audit(directory):
    result=constructed(adaptive_required=True)(directory)
    original.need(result['same_backend_exact_output_qualified']is True,'all3 complete463 own AR outputs exactly equal')
    return dict(result,actual_per_output_geometry_audited=True,original_input_mask_slot_page_canary_Native_and_output_checks_retained=True,
                production_tokens_used_as_reference_inputs=False)
