"""One common CPU status at the original draft/proposal boundary.

No collective runs inside draft-only callbacks. Failure is held until every
rank finishes the same role-local stage, then all ranks raise into the original
fatal cleanup. Modified KV is never reused through catch-and-fallback.
"""
import ast,copy,traceback
from scope159 import require

def capture(engine,error):
    # Preserve the first cause, including local lease/Native failures. No result
    # is fabricated or published, and the current cache is not used again.
    if getattr(engine,'_v159_draft_error',None) is None:
        engine._v159_draft_error=dict(rank=engine.rank,type=type(error).__name__,
            message=str(error),traceback=''.join(traceback.format_exception(type(error),error,error.__traceback__)))

def common(engine):
    import torch
    import torch.distributed as dist
    from host_tree_transport import group,count
    error=getattr(engine,'_v159_draft_error',None)
    status=torch.tensor([int(error is not None)],dtype=torch.int32,device='cpu')
    dist.all_reduce(status,op=dist.ReduceOp.MAX,group=group(engine))
    count(engine,'draft_status')
    if int(status.item()):
        raise RuntimeError('V159 common draft failure before proposal publication: '+
            ('peer draft stage failed' if error is None else repr(error)))

def rewrite(source):
    """Exactly two role-local try blocks and one all-rank common status call.

    Current target execution stays between the two blocks. Original arrival,
    target submission, verdict, publication, clocks and fatal handler survive
    an exact AST inverse. This consumes fully built V148 typed live source.
    """
    tree=ast.parse(source);original=copy.deepcopy(tree)
    found=dict(submit=0,post=0,status=0)
    submit_start="with torch.profiler.record_function('SpecSLO/DraftCompute'):"
    submit_end='target_residual_started = time.perf_counter()'
    post_start="local_rows = None if draft_output is None else draft_output['draft_token_ids']"
    post_end="profile_detail_end(draft_postprocess_profile_started, 'draft_postprocess')"
    status_anchor='exchange_started = time.perf_counter()'
    def wrap(body,start,end,name,*,inclusive=False,clear=False):
        starts=[i for i,n in enumerate(body) if ast.unparse(n).startswith(start)]
        if not starts:return
        require(len(starts)==1,'one exact '+name+' start');a=starts[0]
        ends=[i for i in range(a+1,len(body)) if ast.unparse(body[i])==end]
        require(len(ends)==1,'one exact '+name+' end');b=ends[0]+int(inclusive)
        handler=ast.parse("""try:
 pass
except BaseException as _v159_error:
 _v159_capture_draft_error(self,_v159_error)
 draft_output=None
 local_rows=None
 selected_indices=None
 confidences=None
""").body[0]
        handler.body=body[a:b]
        marker=ast.parse("self._v159_draft_error=None").body if clear else []
        body[a:b]=marker+[handler];found[name]+=1
    def visit(node):
        for field,value in ast.iter_fields(node):
            if isinstance(value,ast.AST):visit(value)
            elif isinstance(value,list):
                for child in list(value):
                    if isinstance(child,ast.AST):visit(child)
                if value and all(isinstance(n,ast.stmt) for n in value):
                    wrap(value,submit_start,submit_end,'submit',clear=True)
                    wrap(value,post_start,post_end,'post',inclusive=True)
                    slots=[i for i,n in enumerate(value) if ast.unparse(n)==status_anchor]
                    if slots:
                        require(len(slots)==1,'single original proposal boundary')
                        value.insert(slots[0],ast.parse('_v159_common_draft_status(self)').body[0]);found['status']+=1
    visit(tree)
    require(found==dict(submit=1,post=1,status=1),'exact live draft status anchors')
    # Reverse only the three marked additions; no generic exception rewriting.
    reverse=copy.deepcopy(tree)
    def undo(node):
        for field,value in ast.iter_fields(node):
            if isinstance(value,ast.AST):undo(value)
            elif isinstance(value,list):
                for child in list(value):
                    if isinstance(child,ast.AST):undo(child)
                if not value or not all(isinstance(n,ast.stmt) for n in value):continue
                out=[]
                for item in value:
                    if ast.unparse(item) in ('self._v159_draft_error = None','_v159_common_draft_status(self)'):continue
                    if isinstance(item,ast.Try) and len(item.handlers)==1 and item.handlers[0].name=='_v159_error':out.extend(item.body)
                    else:out.append(item)
                value[:]=out
    undo(reverse)
    require(ast.dump(reverse,include_attributes=False)==ast.dump(original,include_attributes=False),'entire live status AST inverse')
    return ast.unparse(ast.fix_missing_locations(tree))+'\n',dict(counts=found,exact_AST_inverse=True,
        added_common_Gloo_allreduce_per_draft_stage=1,role_local_collectives_added=0,
        old_proposal_wire_unchanged=True,no_successful_result_after_error=True)

def install(native,live_decode_patch):
    """After original tree_entry transport install, before live decode install."""
    require('_v159_common_draft_status' not in vars(native),'one status installation')
    native._v159_capture_draft_error=capture;native._v159_common_draft_status=common
    original=live_decode_patch.build_live_runtime
    def builder(source=None):return rewrite(original(source))[0]
    live_decode_patch.build_live_runtime=builder
    return dict(original_builder=original,new_builder=builder)
