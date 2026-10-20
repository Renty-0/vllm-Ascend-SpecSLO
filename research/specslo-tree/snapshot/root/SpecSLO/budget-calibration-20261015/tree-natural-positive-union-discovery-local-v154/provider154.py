"""Only actual input prefixes are exposed; no observed future target tokens."""
import json
from scope154 import P,require,digest
def fixture():return json.loads((P/'FIXTURE_INPUT154.json').read_text())
def source_contexts(case):
 from originals151 import original_function
 original=original_function('short_scope_v126','source_contexts')(case)
 require(len(original)==64 and case.prefix==128,'exact original source scope')
 out=[dict(row,token_ids=list(row['token_ids']),V154_source_kind='original-P151-negative-context') for row in original]
 candidates=fixture()['real_source_prefixes'];require(len(candidates)==8,'eight retained actual prefixes')
 for owner in range(32,48):
  row=candidates[(owner-32)%8];tokens=row['source_token_ids_through_current_root']
  require(tokens==row['prompt_token_ids']+row['actual_previous_completion_token_ids'] and len(tokens)==129,
   'only original prompt plus actually emitted previous completion through current root')
  out[owner]=dict(token_ids=list(tokens),context_length=129,category='retained-full463-prefix',
   V154_source_kind='retained-full463-p128-input-prefix',source_request_id=row['request_id'],
   source_request_index=row['request_index'],source_target_call=row['target_call_index'],source_stream_event=row['stream_event_index'])
 return out
def evidence(case):
 contexts=source_contexts(case)
 return dict(real_candidate_owners=list(range(32,48)),old_negative_owners=list(range(48,64)),
  retained_candidate_request_identities=8,retained_unique_token_prefixes=len({tuple(row['source_token_ids_through_current_root']) for row in fixture()['real_source_prefixes']}),
  actual_input_prefix_sha256={str(i):digest(v['token_ids'][:129]) for i,v in enumerate(contexts)},
  source_kinds={str(i):v['V154_source_kind'] for i,v in enumerate(contexts)},
  post_root_observed_target_tokens_exposed=False,old_promotion_or_new_positive_claimed=False)
