"""Deterministic bounded FIA extents; preserve exact real query/KV lengths.

The whitelist is frozen before startup. Native cache contents never select a
bucket, and the original capture/validation/fallback gates remain authoritative.
"""
import json
from pathlib import Path

TOKENS=(40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,116,120,124,128,165,197)
SEGMENTS=(4,8,10,12,14,16,17,18,22,24,32,33,34,38)
_MANIFEST=json.loads((Path(__file__).resolve().parent/'BOUNDED_BUCKET_WHITELIST.json').read_text())
STATIC_NATIVE_KEYS=frozenset(tuple(row) for row in _MANIFEST['static_native_keys'])


def _lengths(real_count, real_queries, tokens, segments):
    dummy=segments-real_count
    extra=tokens-real_queries
    if dummy==0 and extra==0:
        return []
    if not 1<=dummy<=32 or not 2*dummy<=extra<=9*dummy:
        return None
    return [extra//dummy+int(i<extra%dummy) for i in range(dummy)]


def legacy_envelope(real_count, real_queries):
    """Original fixed MC2 envelope, independent of native cache contents."""
    if not 1<=real_count<=32 or not 2*real_count<=real_queries<=128:
        raise ValueError('Real target requests/queries exceed the service envelope')
    for tokens in TOKENS:
        for segments in SEGMENTS:
            lengths=_lengths(real_count,real_queries,tokens,segments)
            if lengths is not None:
                return tokens,segments,lengths
    raise ValueError('No physical graph can absorb exact real queries')


def envelope(real_count, real_queries, family):
    """Map admitted B1..5 geometries to a statically bounded native key set."""
    if family not in ('causal','full9'):
        raise ValueError('Target family must describe the actual real plans')
    if (not 1<=real_count<=32
            or not 2*real_count<=real_queries<=min(6*real_count,real_count+96,128)
            or (family=='causal' and real_queries>5*real_count)
            or (family=='full9' and real_queries<2*real_count+4)):
        raise ValueError('Target geometry is outside the admitted budget/family domain')
    original=legacy_envelope(real_count,real_queries)
    if (original[0],original[1],family) in STATIC_NATIVE_KEYS:
        return original
    candidates=((96,32),) if real_count<=5 else ((96,38),(128,38),(165,38))
    for tokens,segments in candidates:
        lengths=_lengths(real_count,real_queries,tokens,segments)
        if lengths is not None:
            return tokens,segments,lengths
    raise ValueError('No admitted bounded bucket preserves the real target geometry')


def plan_family(plans):
    """Use the original model's structural predicate on real plans only."""
    from vllm_ascend.spec_decode.pearl.native_model import _is_contiguous_linear_tree_plan
    linear=[]
    for plan in plans:
        budget=int(plan.candidate_budget)
        contiguous=bool(_is_contiguous_linear_tree_plan(plan))
        if not 1<=budget<=5 or contiguous!=(budget<=4):
            raise ValueError('Bounded qualification admits causal B1..4 and genuinely branched B5 only')
        linear.append(contiguous)
    if not linear:
        raise ValueError('A target family requires at least one real plan')
    return 'causal' if all(linear) else 'full9'


def pad_target(engine,plans,roots,rows,ids,active,states):
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    count=len(plans)
    if not count:
        return list(plans),list(roots),list(rows),list(ids),count
    if not count==len(roots)==len(rows)==len(ids) or len(set(ids))!=count:
        raise ValueError('Real target requests must be aligned and independent')
    family=plan_family(plans)
    real_queries=sum(p.candidate_budget+1 for p in plans)
    tokens,segments,lengths=envelope(count,real_queries,family)
    scratch=engine._tree_graph_scratch_sequence_ids
    selected_scratch=scratch[:len(lengths)]
    if (len(selected_scratch)!=len(lengths) or len(set(selected_scratch))!=len(lengths)
            or set(selected_scratch)&set(ids)):
        raise RuntimeError('Not enough independent dedicated target scratch sequences')
    pp,rr,dd,ss=list(plans),list(roots),list(rows),list(ids)
    for seq,n in zip(selected_scratch,lengths):
        pp.append(cached_cpu_tree_speculation_plan(1,n-1,0,
            engine.config.max_model_len,candidate_budget=n-1))
        rr.append(0);dd.append([0]*(n-1));ss.append(seq)
    assert sum(p.candidate_budget+1 for p in pp)==tokens and len(pp)==segments
    return pp,rr,dd,ss,count
