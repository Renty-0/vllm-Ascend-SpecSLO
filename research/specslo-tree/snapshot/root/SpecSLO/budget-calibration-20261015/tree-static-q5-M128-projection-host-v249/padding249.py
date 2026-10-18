"""Retain real proposals; fill128 physical rows with private original dummy plans."""
def pad(engine,plans,roots,rows,ids,active,states):
    from mapper249 import envelope
    from target_layout import plan_family
    from padding231 import root_only
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    n=len(plans)
    if not n==len(roots)==len(rows)==len(ids)or len(set(ids))!=n:raise ValueError('exact original real proposal alignment')
    m,s,lengths=envelope(n,sum(p.candidate_budget+1 for p in plans),plan_family(plans))
    scratch=engine._tree_graph_scratch_sequence_ids[:len(lengths)]
    if len(scratch)!=len(lengths)or len(set(scratch))!=len(scratch)or set(scratch)&set(ids):raise ValueError('independent original private dummy owners')
    pp,rr,dd,ss=list(plans),list(roots),list(rows),list(ids)
    for owner,q in zip(scratch,lengths,strict=True):
        pp.append(root_only(engine.config.max_model_len)if q==1 else cached_cpu_tree_speculation_plan(1,q-1,0,engine.config.max_model_len,candidate_budget=q-1))
        rr.append(0);dd.append([0]*(q-1));ss.append(owner)
    if len(pp)!=s or sum(p.candidate_budget+1 for p in pp)!=128:raise RuntimeError('all actual physical128 rows accounted')
    return pp,rr,dd,ss,n
