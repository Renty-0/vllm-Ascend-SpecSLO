"""Private clone of the original dispatcher: append only the copy-grid key.

The Native capture, changed-input validation, replay and failure bodies remain
unchanged. Each captured model graph owns one constant copy program count.
"""
import ast,copy,inspect,textwrap,types

def projected(function):
    node=ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
    before=ast.dump(node,include_attributes=False)
    assignments=[n for n in ast.walk(node)if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='entry_key']
    if len(assignments)!=1:raise RuntimeError('ROOT279 one original actual target capture-key declaration')
    item=assignments[0];old=copy.deepcopy(item.value)
    item.value=ast.Call(func=ast.Name(id='_ROOT279_checked_capture_key',ctx=ast.Load()),
        args=[ast.Name(id='self',ctx=ast.Load()),ast.Name(id='attention_metadatas',ctx=ast.Load()),copy.deepcopy(old)],keywords=[])
    source=ast.unparse(ast.fix_missing_locations(node))+'\n';item.value=old
    if ast.dump(node,include_attributes=False)!=before:raise RuntimeError('ROOT279 whole original Native dispatcher AST inverse')
    return source

def key_for(engine,runner,metadata,key):
    if runner is not engine.graph_runner:raise RuntimeError('ROOT279 actual runner instance changed')
    state=engine._branch_tail_state;current=state.static_current
    if current is None or len(metadata)!=1 or metadata[0]is not current['physical'][2]:return key
    if not current['copy_enabled']:
        if state.enabled:raise RuntimeError('ROOT279 inactive physical copy has active layer state')
        state.copy279_last_dispatch_key=repr(key)
        return key
    grid=state.copy279_bucket
    if grid not in(128,256,512,1024,1792)or not state.enabled:
        raise RuntimeError('ROOT279 captured branch needs one validated positive grid')
    result=(key[0]+'|ROOT279:copy-grid:'+str(grid),key[1])
    state.copy279_last_dispatch_key=repr(result)
    return result

def install(engine):
    runner=engine.graph_runner;original=runner._run_target
    function=original.__func__
    if function.__name__!='_run_target':raise RuntimeError('ROOT279 original Native target dispatcher required')
    source=projected(function)
    def checked(runner,metadata,key):return key_for(engine,runner,metadata,key)
    namespace=dict(function.__globals__,_ROOT279_checked_capture_key=checked)
    exec(compile('from __future__ import annotations\n'+source,__file__+'::single-copy-grid-key-extension','exec'),namespace)
    patched=namespace['_run_target']
    if inspect.signature(patched)!=inspect.signature(function):raise RuntimeError('ROOT279 original Native dispatcher signature')
    bound=types.MethodType(patched,runner);runner._run_target=bound
    runner._ROOT279_original_target_dispatch=function
    runner._ROOT279_private_target_dispatch=bound
    return dict(original_body_AST_inverse=True,only_copy_grid_key_extended=True,
        Native_capture_and_changed_input_replay_bodies_preserved=True,Native_flags_assigned=False)
