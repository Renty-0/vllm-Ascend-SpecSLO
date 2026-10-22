"""Genuine original prefixes only; remap candidate identities to parent owners."""
from types import SimpleNamespace
from originals151 import original_function
from provider199 import source_contexts as prior
from scope419 import require


def source_contexts(case):
    original=original_function('short_scope_v126','source_contexts')(SimpleNamespace(count=64,prefix=128,repair=1))
    selected=prior(case)
    rows=[dict(v,token_ids=list(v['token_ids']))for v in original]
    for i in range(8):rows[24+i]=dict(selected[16+i],token_ids=list(selected[16+i]['token_ids']))
    for i in range(16):rows[32+i]=dict(original[48+i],token_ids=list(original[48+i]['token_ids']))
    require(len(rows)==64 and all(len(v['token_ids'])>=129 for v in rows),'all64 genuine through-root source prefixes')
    return rows


def evidence(case):
    from scope419 import digest
    rows=source_contexts(case)
    return dict(real_candidate_owners=list(range(24,32)),old_negative_owners=list(range(32,48)),
        actual_candidate_request_identities=8,
        unique_candidate_token_prefixes=len({tuple(rows[i]['token_ids'][:129])for i in range(24,32)}),
        source_prefix_sha256={str(i):digest(row['token_ids'][:129])for i,row in enumerate(rows)},
        post_root_observed_target_tokens_exposed=False,old_raw_accepted_path_available=False,
        new_natural_positive_assumed=False)
