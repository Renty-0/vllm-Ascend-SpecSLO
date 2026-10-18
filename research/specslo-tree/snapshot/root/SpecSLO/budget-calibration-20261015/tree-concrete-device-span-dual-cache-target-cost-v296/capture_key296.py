"""Private clone of the original dispatcher: append only the copy-grid key.

The Native capture, changed-input validation, replay and failure bodies remain
unchanged. Each captured model graph owns one constant copy program count.
"""
import ast,copy,inspect,textwrap,types,__future__
from pathlib import Path

def actual_source(function):
    """Recover the actual retained replay rewrite, never inspect generated code."""
    import tree_target_replay as replay
    path=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_graph.py')
    text=path.read_text();node=next(n for n in ast.walk(ast.parse(text))
        if isinstance(n,ast.FunctionDef)and n.name=='_run_target')
    original='\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])+'\n'
    source=replay.patched_source(original)
    filename=str(Path(replay.__file__))+':_run_target'
    if (function.__code__.co_filename!=filename
            or function.__globals__.get('_dynamic_tree_replay_submit')is not replay.submit):
        raise RuntimeError('ROOT296 actual retained replay-first dispatcher provenance')
    namespace=dict(function.__globals__)
    flags=function.__code__.co_flags&__future__.annotations.compiler_flag
    exec(compile(source,filename,'exec',flags=flags,dont_inherit=True),namespace)
    expected=namespace['_run_target']
    if expected.__code__!=function.__code__ or inspect.signature(expected)!=inspect.signature(function):
        raise RuntimeError('ROOT296 regenerated retained replay body or signature differs from actual function')
    return source

def projected(function):
    node=ast.parse(actual_source(function)).body[0]
    before=ast.dump(node,include_attributes=False)
    assignments=[n for n in ast.walk(node)if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='entry_key']
    if len(assignments)!=1:raise RuntimeError('ROOT296 one original actual target capture-key declaration')
    item=assignments[0];old=copy.deepcopy(item.value)
    item.value=ast.Call(func=ast.Name(id='_ROOT296_checked_capture_key',ctx=ast.Load()),
        args=[ast.Name(id='self',ctx=ast.Load()),ast.Name(id='attention_metadatas',ctx=ast.Load()),copy.deepcopy(old)],keywords=[])
    source=ast.unparse(ast.fix_missing_locations(node))+'\n';item.value=old
    if ast.dump(node,include_attributes=False)!=before:raise RuntimeError('ROOT296 whole original Native dispatcher AST inverse')
    return source

def key_for(engine,runner,metadata,key):
    if runner is not engine.graph_runner:raise RuntimeError('ROOT296 actual runner instance changed')
    state=engine._branch_tail_state;current=state.static_current
    if current is None or len(metadata)!=1 or metadata[0]is not current['physical'][2]:return key
    if not current['copy_enabled']:
        if state.enabled:raise RuntimeError('ROOT296 inactive physical copy has active layer state')
        state.copy280_last_dispatch_key=repr(key)
        return key
    grid=state.copy280_bucket
    if grid not in(128,256,512,1024,1792)or not state.enabled:
        raise RuntimeError('ROOT296 captured branch needs one validated positive grid')
    result=(key[0]+'|ROOT296:copy-grid:'+str(grid),key[1])
    state.copy280_last_dispatch_key=repr(result)
    return result

def install(engine):
    runner=engine.graph_runner;original=runner._run_target
    function=original.__func__
    if function.__name__!='_run_target':raise RuntimeError('ROOT296 original Native target dispatcher required')
    source=projected(function)
    def checked(runner,metadata,key):return key_for(engine,runner,metadata,key)
    namespace=dict(function.__globals__,_ROOT296_checked_capture_key=checked)
    flags=function.__code__.co_flags&__future__.annotations.compiler_flag
    exec(compile(source,__file__+'::single-copy-grid-key-extension','exec',
        flags=flags,dont_inherit=True),namespace)
    patched=namespace['_run_target']
    if inspect.signature(patched)!=inspect.signature(function):raise RuntimeError('ROOT296 original Native dispatcher signature')
    bound=types.MethodType(patched,runner);runner._run_target=bound
    runner._ROOT296_original_target_dispatch=function
    runner._ROOT296_private_target_dispatch=bound
    return dict(original_body_AST_inverse=True,only_copy_grid_key_extended=True,
        Native_capture_and_changed_input_replay_bodies_preserved=True,Native_flags_assigned=False)
