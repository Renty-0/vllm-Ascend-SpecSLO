"""Expose only original prompts plus actual old completion through current root."""
import json
from types import SimpleNamespace
from base199 import P,require
from scope199 import digest

def source_contexts(case):
 from originals151 import original_function
 original=original_function('short_scope_v126','source_contexts')(SimpleNamespace(count=64,prefix=128,repair=1))
 require(len(original)==64,'original64 genuine dataset source rows')
 out=[dict(v,token_ids=list(v['token_ids']),V199_source_kind='original-P151-source') for v in original]
 rows=json.loads((P/'FIXTURE_INPUT199.json').read_text())['real_source_prefixes']
 require(len(rows)==8,'eight actual retained prefix candidates')
 for i,row in enumerate(rows):
  tokens=list(row['source_token_ids_through_current_root'])
  require(tokens==row['prompt_token_ids']+row['actual_previous_completion_token_ids'] and len(tokens)==129,
   'no observed post-root token enters actual candidate inputs')
  out[16+i]=dict(token_ids=tokens,V199_source_kind='P166-full32-retained-p128-B4-stream5-candidate',
   source_request_id=row['request_id'],source_request_index=row['request_index'],source_target_call=row['source_target_call'])
 for i in range(8):out[24+i]=dict(original[48+i],token_ids=list(original[48+i]['token_ids']),V199_source_kind='original-P151-natural-negative-candidate')
 return out

def evidence(case):
 rows=source_contexts(case);fixture=json.loads((P/'FIXTURE_INPUT199.json').read_text())
 return dict(real_candidate_owners=list(range(16,24)),old_negative_owners=list(range(24,32)),
  actual_candidate_request_identities=8,unique_candidate_token_prefixes=len({tuple(r['source_token_ids_through_current_root']) for r in fixture['real_source_prefixes']}),
  source_prefix_sha256={str(i):digest(r['token_ids'][:129]) for i,r in enumerate(rows)},
  source_kinds={str(i):v['V199_source_kind'] for i,v in enumerate(rows)},post_root_observed_target_tokens_exposed=False,
  old_raw_accepted_path_available=False,new_natural_positive_assumed=False)
