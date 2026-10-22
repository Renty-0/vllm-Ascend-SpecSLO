"""Original Native preflight/vote/compaction and promotion statements.

Compile exact AST statement ranges, after the unchanged F16 canonical rewrite.
No acceptance predicate, consensus lane or compaction rule is reimplemented.
The adapter supplies single-round locals and performs no timing or streaming.
"""
import ast
import copy
import hashlib
import inspect
import textwrap


def assigned(node, name):
    return (isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
        or isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id==name)


def compile_protocol(native):
    import canonical_eager
    original=textwrap.dedent(inspect.getsource(native.NativePearlEngine._generate_spec_rhythm_tree_decode))
    source=canonical_eager.patch_decode_source(original)
    import host_tree_transport
    source=host_tree_transport.patch_decode_source(source)
    tree=ast.parse(source)
    bodies=[node.body for node in ast.walk(tree) if hasattr(node,'body') and isinstance(node.body,list)]
    main=next(body for body in bodies if any(assigned(node,'preflight_error') for node in body))
    start=next(i for i,node in enumerate(main) if assigned(node,'preflight_error'))
    preflight=main[start:start+3]
    assert isinstance(preflight[2],ast.Try)
    status=next(i for i,node in enumerate(main) if assigned(node,'preflight_status'))
    consensus=main[status:status+3]
    assert isinstance(consensus[1],ast.Assign) and ast.unparse(consensus[1].value.func)=='_dt_cpu_reduce'
    failure=next(node for node in main[status+3:] if isinstance(node,ast.If)
        and 'structural_failed' in ast.unparse(node.test) and 'numerical_failed' in ast.unparse(node.test))
    compaction_start=next(i for i,node in enumerate(main) if isinstance(node,ast.Expr)
        and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='cache_mappings.update')
    compaction_end=next(i for i,node in enumerate(main[compaction_start:],compaction_start)
        if isinstance(node,ast.If) and ast.unparse(node.test)=='rows_to_compact')
    compaction=main[compaction_start:compaction_end+1]
    commit_body=next(body for body in bodies if any(assigned(node,'accepted_count') for node in body)
        and any(assigned(node,'promoted') for node in body))
    first=next(i for i,node in enumerate(commit_body) if assigned(node,'accepted_count'))
    last=next(i for i,node in enumerate(commit_body[first:],first) if assigned(node,'promoted'))
    promotion=commit_body[first:last+1]
    cleanup_start=next(i for i,node in enumerate(commit_body) if isinstance(node,ast.Expr)
        and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='payloads.pop')
    cleanup=commit_body[cleanup_start:cleanup_start+3]
    assert len(cleanup)==3 and isinstance(cleanup[-1],ast.If)
    # Do not permit an AST anchor to silently omit any original guard.
    body_text=ast.unparse(ast.Module(body=promotion,type_ignores=[]))
    for required in ('observed_path == dependency_tokens','observed_frontier == int(eager_frontier)',
                     'controller.validate_verification(index)','controller.finish_verification'):
        if required not in body_text:raise RuntimeError('Original promotion guard disappeared: '+required)
    skeleton=ast.parse('def execute():\n    pass\n').body[0]
    names=('self','controller','payloads','cache_mappings','target_indices','target_plans','target_payloads',
        'target_rows','target_output','local_states','output_rows','accepted_rows','counters')
    skeleton.args.args=[ast.arg(arg=name) for name in names]
    skeleton.body=copy.deepcopy(preflight+consensus+[failure]+compaction)+ast.parse('return preflight_host_cache_mappings').body
    namespace=dict(vars(native))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[skeleton],type_ignores=[])),__file__+':original-preflight-consensus-compact','exec'),namespace)
    execute=namespace['execute']
    function=ast.parse('def finish():\n    pass\n').body[0]
    args=('self','controller','index','tree_plan','target_payloads','payloads','output_rows','accepted_rows',
          'committed_tokens','will_finish','counters')
    function.args.args=[ast.arg(arg=name) for name in args]
    function.body=ast.parse('row=0').body+copy.deepcopy(promotion)+ast.parse('return promoted').body
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),__file__+':original-promotion','exec'),namespace)
    release=ast.parse('def cleanup():\n    pass\n').body[0]
    release.args.args=[ast.arg(arg=name) for name in ('self','index','payloads','cache_mappings','current_ticket','eager_ticket','promoted','counters')]
    release.body=ast.parse('promoted_cache_rows=[]').body+copy.deepcopy(cleanup)+ast.parse('return promoted_cache_rows').body
    exec(compile(ast.fix_missing_locations(ast.Module(body=[release],type_ignores=[])),__file__+':original-payload-cleanup','exec'),namespace)
    return dict(preflight_consensus_compact=execute,finish=namespace['finish'],cleanup=namespace['cleanup'],
        original_decode_sha256=hashlib.sha256(original.encode()).hexdigest(),
        canonical_decode_sha256=hashlib.sha256(source.encode()).hexdigest(),
        extracted_structural_and_numerical_lanes_unchanged=True)


def commit_states(draft_state,target_state,output,accepted,proposed):
    """Original replicated sequence-state method; no draft KV claim changes."""
    count=sum(node>=0 for node in accepted)
    values=[int(token) for token in output if token>=0]
    for state in (draft_state,target_state):
        state.apply_tree_verification(output_token_ids=values,accepted=count,proposed_tokens=proposed)
    return values
