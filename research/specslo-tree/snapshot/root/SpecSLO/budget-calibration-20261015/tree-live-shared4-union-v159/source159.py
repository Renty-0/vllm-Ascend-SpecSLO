"""Exact source overlay; no installation, backend import or frozen writes."""
import ast,hashlib,importlib.util,json,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent
BASE=P.parent/'tree-high-normal-cap4-native-families-v148'
PROTO=P.parent/'tree-union-shared4-protocol-local-v152'
FUSION=P.parent/'budget-diagnostics/draft-repair-frontier-fusion-v149-source/repair_frontier_fusion.py'

def checked(path):
    path=Path(path);wanted=json.loads((P/'INPUTS159.json').read_text())['readonly'][str(path)]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=wanted:raise RuntimeError('V159 readonly source changed: '+str(path))
    return path

def module(name,path):
    path=checked(path);old=sys.modules.get(name)
    if old is not None:
        if Path(old.__file__).resolve()!=path:raise RuntimeError('V159 canonical origin occupied: '+name)
        return old
    spec=importlib.util.spec_from_file_location(name,path);out=importlib.util.module_from_spec(spec)
    sys.modules[name]=out
    try:spec.loader.exec_module(out)
    except BaseException:sys.modules.pop(name,None);raise
    return out

def fast_source():
    original=checked(PROTO/'fast_tree_draft_v5.py').read_text()
    begin=original.index('        # Prototype endpoint opt-in only.')
    end=original.index('        repair_calls=0\n',begin)
    old=original[begin:end]
    new='''        # Same normal4/parent3/child3 branch; known-token checks above and
        # all original ledger/publication code below remain authoritative.
        from live159 import fused_prelude
        fused_plan,predicted=fused_prelude(engine,native,runner_cls,
            plans,ids,sources,repairs,frontiers,steps)
'''
    result=original[:begin]+new+original[end:]
    if result.replace(new,old)!=original:raise RuntimeError('V159 exact prelude delegate inverse differs')
    # Entire parent/source/catchup validation, all ledger updates, main chain,
    # canonical raw8/confidence and query counters stay as P152's tested source.
    ast.parse(result)
    return result,dict(parent=str(PROTO/'fast_tree_draft_v5.py'),removed=old,added=new,
        exact_inverse=True,Native_flags_assigned=False)

def shape_namespace(capability):
    source=checked(BASE/'split_shape_v132.py').read_text()
    old='    shared_steps4=bool(plan.normal_budgets) and max(plan.normal_budgets.values())>3\n'
    new=old+'''    shared4_ready=(shared_steps4 and max(plan.normal_budgets.values())==4
        and _v159_capability.allows_shared4(len(plan.normal_budgets),sum(b==3 for b in plan.eager_budgets.values())))
'''
    replacements=((old,new),
        ('if shared_steps4 else 0','if shared_steps4 and not shared4_ready else 0'),
        ('if b==3 and not shared_steps4','if b==3 and (not shared_steps4 or shared4_ready)'))
    counts=[]
    for before,after in replacements:
        count=source.count(before);expected=2 if before=='if shared_steps4 else 0' else 1
        if count!=expected:raise RuntimeError('V159 exact shape anchor count differs')
        source=source.replace(before,after);counts.append(count)
    reverse=source
    for before,after in reversed(replacements):reverse=reverse.replace(after,before)
    if reverse!=checked(BASE/'split_shape_v132.py').read_text():raise RuntimeError('V159 shape inverse differs')
    namespace=dict(__name__='specslo_v159_shape',__file__=str(P/'shape159.generated.py'),_v159_capability=capability)
    exec(compile(source,str(P/'shape159.generated.py'),'exec'),namespace)
    namespace['_source_contract']=dict(exact_inverse=True,counts=counts,parent_B3_filter_unchanged=True,
        normal_cap_threshold_8_unchanged=True,normal_B5_still_defers_eager=True)
    return namespace,source

def controls_namespace(shape):
    """Original CandidateControls class with exactly its shape delegate rebound.

    Call this factory instead of importing the frozen class in the new worker.
    Its original shape identity/CPU/config/mapper/rollback checks remain intact.
    """
    source=checked(BASE/'candidate_controls_v132.py').read_text()
    old='from split_shape_v132 import shape,filter_eager_parents,refine_draft_groups'
    if source.count(old)!=1:raise RuntimeError('V159 controls import anchor differs')
    changed=source.replace(old,'shape=_v159_shape\nfilter_eager_parents=_v159_filter\nrefine_draft_groups=_v159_refine')
    anchor='P=Path(__file__).resolve().parent'
    if changed.count(anchor)!=1:raise RuntimeError('V159 controls readonly root anchor differs')
    changed=changed.replace(anchor,'P=BASE')
    namespace=dict(__name__='specslo_v159_controls',__file__=str(P/'controls159.generated.py'),BASE=BASE,
        _v159_shape=shape['shape'],_v159_filter=shape['filter_eager_parents'],_v159_refine=shape['refine_draft_groups'])
    exec(compile(changed,str(P/'controls159.generated.py'),'exec'),namespace)
    namespace['_v159_generated_source']=changed
    return namespace

def tree_entry_namespace(output_directory):
    """Read original JSON/source pins, direct every original write to new epoch.

    This leaves original Native inventories and M197 prefill geometry unchanged.
    New worker must invoke prepare_helpers BEFORE this install(), then own new
    components BEFORE sealed formal warm/full; this is not a full launcher.
    """
    output=Path(output_directory).resolve()
    if output!=P:raise RuntimeError('V159 factory writes only its own private epoch')
    source=checked(BASE/'tree_entry.py').read_text()
    edits=(("P = Path(__file__).resolve().parent","P = BASE"),
        ("(P/'runtime_tree_decode.py').write_text(source)","(OUTPUT/'runtime_tree_decode.py').write_text(source)"),
        ("str(P/'runtime_tree_decode.py')","str(OUTPUT/'runtime_tree_decode.py')"),
        ("progress_dir=P/'generation-progress'","progress_dir=OUTPUT/'generation-progress'"))
    for old,new in edits:
        if source.count(old)!=1:raise RuntimeError('V159 entry writable anchor differs')
        source=source.replace(old,new)
    namespace=dict(__name__='specslo_v159_tree_entry',__file__=str(P/'tree_entry159.generated.py'),BASE=BASE,OUTPUT=output)
    exec(compile(source,str(P/'tree_entry159.generated.py'),'exec'),namespace)
    namespace['_v159_generated_source']=source
    return namespace

def artifacts():
    """HOST constructor for explicit source files; no install or writes."""
    from scope159 import Capability
    shape,shape_source=shape_namespace(Capability())
    return {'fast159.generated.py':fast_source()[0],
        'shape159.generated.py':shape_source,
        'controls159.generated.py':controls_namespace(shape)['_v159_generated_source'],
        'tree_entry159.generated.py':tree_entry_namespace(P)['_v159_generated_source'],
        # Original install re-emits exactly this pinned source before compiling
        # it. Prepare it now so the future freeze includes its actual bytes.
        'runtime_tree_decode.py':checked(BASE/'runtime_tree_decode.py').read_text()}

def emit_artifacts():
    """HOST preparation only; actual runtime verifies these frozen bytes."""
    result={}
    for name,source in artifacts().items():
        path=P/name
        with path.open('x') as stream:stream.write(source)
        result[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result

def require_artifacts():
    for name,source in artifacts().items():
        if (P/name).read_text()!=source:raise RuntimeError('V159 generated source pin differs: '+name)
