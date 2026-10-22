from pathlib import Path
import hashlib,json,datetime

B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
P=B/'tree-empty-active-cohort-causal80-active32-full-v197'
O=B/'tree-high-normal-cap4-native-families-v148'
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
proposal=B/'tree-b4-four-query-no-bonus-cohort16-full-v200'
assert not proposal.exists()
paths=[D/'TREE_B4_NO_BONUS_FOUR_QUERY_FIXED_WINDOW_MINIMUM_SOURCE_FEASIBILITY_REVISION2.json',
 P/'FROZEN_SOURCE197.json',P/'INPUTS197.json',P/'runtime197.py',P/'extension197.py',P/'source197.py',
 P/'scope197.py',P/'empty197.py',P/'component197.py',P/'prefill197.py',P/'next4_197.py',P/'audit197.py',
 O/'original_native_protocol_v75.py',O/'original_tree_cycle_v75.py',O/'runtime_tree_decode.py',
 O/'tree_entry.py',O/'host_tree_transport.py',O/'host_target_metadata.py',O/'split_shape_v132.py',
 O/'split_budget_v132.py',O/'budget_policy.py',O/'target_layout.py',O/'complete_search.py',
 O/'EXACT_QUALIFIED_KEY_INVENTORY.json',O/'BOUNDED_BUCKET_WHITELIST.json',O/'finite_inventory_plan.py',
 O/'startup_receipts_v75.py',O/'fast_tree_draft_v5.py',O/'canonical_eager.py',O/'root_seed.py',
 N/'native_engine.py',N/'native_model.py',N/'native_graph.py',N/'spec_rhythm.py',
 N.parents[1]/'ops/triton/spec_decode/fixed_greedy_verdict.py']
pins={str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in paths}
old44=json.loads((O/'EXACT_QUALIFIED_KEY_INVENTORY.json').read_text())['geometries']
static=json.loads((O/'BOUNDED_BUCKET_WHITELIST.json').read_text())['static_native_keys']
assert len(old44)==44 and [64,16,'causal'] not in old44
assert [64,16,'causal'] not in static
tail='|steps:1|fia:exact-length-task-update|buffers:(((16, 32), (16, 32), (2048, 2048)),)'
out={
 'schema':'V200.minimum-four-query-no-BONUS-implementation-plan.v1',
 'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'status':'source_only_plan_awaiting_explicit_implementation_authorization',
 'candidate_directory_proposed':str(proposal), 'candidate_directory_exists':False,
 'scope':dict(candidate_source_written=False,model_or_Torch_or_NPU_imported=False,device_access=False,
  current197_actual_or_receipt_read=False,frozen_sources_modified=False,suites_executed=False),
 'sole_production_parent':dict(path=str(P),freeze_sha256=pins[str(P/'FROZEN_SOURCE197.json')]['sha256'],
  baseline='197 empty-active-only admission; unchanged173 cap4/low-N trueB5, active32, config64/queued463/storage527, C96/E96/global192, all463/output256/original arrivals and clocks'),
 'decision':{
  'source_feasible':True,
  'minimum_is_not_only_a_mapper_change':'A new physical64 target call plus new noBONUS verdict conversion AND a count4 private physical-mapping preflight are necessary. Changing only Q/M or pretending B4 is B3 fails the original protocol ABI.',
  'performance_unknown':'Four draft model stages and four top2/head decisions remain. Four target comparisons remain, but maximum delivery falls5 to4. Query reduction and potential later-repair removal must be measured with actual round counts, not multiplied into GP.',
  'parent_197_context_only':'ROOT reports197 terminal complete and visible BONUS13040/118528 tokens,37281 B4 owner-rounds. These are parent-provided, not independently read or verified here, and are not qualification or an ablation.',
  'implementation_order':'New bounded numerical/protocol/calibration component first; only its current-engine issuer and a separately approved route-aware gain provider permit the normal463 interface. No automatic production switch from old receipts.'},
 'exact_lane':{
  'predicate':['Exactly16 real current normal owners, all logical B4, original contiguous spine parents[-1,0,1,2], four true draft candidates; all greedy within shared vocabulary.',
   'No actual E work, no eager current tickets, no staged continuation depending on this parent, no reserved/private padding owners in the real frame.',
   'Current issued proof, original197 and173 component objects, current M64 G/H objects, ticket/proposal IDs, prefix epochs, source generation, physical allocation and lease all match.',
   'Provisional scheduling label is not permission: recheck the complete actual target frame after grants/publication and before model dispatch; any unsupported domain takes the exact original197/173 route before model execution. Never catch a model/numerical failure and fallback.'],
  'preserved_fallback':['Other N; any heterogeneous budgets; B5 and genuine forks; any E; stochastic mode; unissued, cold or unsupported usage. Keep original root_seed/repair path for oldBONUS/B5 transitions.'],
  'query_recipe':'For each owner at current P: input[root,d1,d2,d3], positions[P,P+1,P+2,P+3]; all16 concatenate to64; query partition[4]*16, S16, no padding/dummy.',
  'logical_accounting':'Logical proposal B4/C64 remains. Original conservative admission query reservation Q80 may remain, with separate admission_reserved_queries80/actual_query_count64/logical_written_queries64/padding0. No negative padding, fake B3 plan, fifth mapping or fifth KV write.'},
 'native_domain':{
  'old44_count':len(old44),'old44_M64_geometries':[r for r in old44 if r[0]==64],
  'old44_causal_S16':[r for r in old44 if r[1]==16 and r[2]=='causal'],
  'new_geometry':[64,16,'causal'],
  'genuinely_new_target_entries':[['target-greedy:151936'+tail,[64]],['target-hidden:0'+tail,[64]]],
  'important':'Both G64/S16 and H64/S16 are absent from the original44 and static37. M64 MC2/profile presence is not Native entry qualification. Original44 only has M64/S12/full9. No new kernel/profile or key alias is proposed.',
  'current_gate':'Capture before/after dictionaries and object identities; original flags close naturally through actual F/T/F and independent sameM checks. Add exactly the two observed original target_entries keys; no entries/draft_entries additions. Do not assign flags, copy entries or borrow fixed/180 identities.',
  'backend':'Retain original Native run_target_greedy/run_tree_hidden target-family machinery, FIA causal metadata and MC2/chained-epilogue guards. Keep mixed_target_graph disabled and generic fixed entries unused.'},
 'smallest_source_boundary':{
  'new_core_modules':['lane200.py: exact eligibility/current-frame lease and current-engine issuer',
   'target200.py: original fixed 4query recipe, original host allocation/page/alias checks and NativeMetadata ABI, actual target G/H call and truthful output schema',
   'verdict200.py: original fixed_greedy_full_window_verdict(no bonus), current-device validated 64head/64draft inputs, lifetime-safe result readback, raw-to-original-tree packet conversion',
   'protocol200.py: checked fresh private extraction/dispatch for count4 preflight; original parent protocol preserved separately',
   'gain200.py: new-policy calibration and separately keyed actual online observations, conditional original-allocation wrapper',
   'actor200.py/audit200.py: bounded new usage, independent fresh histories, actual raw metadata/verdict/commit/next evidence, retained-data HOST audit'],
  'thin_parent_projection':'Project197 required private runtime/source/progress/trace/check/scope/extension/guard identity and IO once with exact whole-text inverse plus explicit listed anchors. Reuse parent prefill197/next4_197 mathematical producers and original173 components by pinned read-only imports; do not clone377 sources or binary archives.',
  'production_hot_changes':['One checked target-forward dispatch for actual issued exact lane, leaving the fallback body exact.',
   'One checked leader-verdict dispatch plus framed status failure propagation into existing original all4 control flow.',
   'One preflight-call AST replacement to private count4/original dispatcher; original finite vote, consensus, contiguous compaction predicate, finish, state apply, publication and cleanup remain.',
   'Actual-route observation routing and conditional g provider/reset, not a globalB4 posterior overwrite.'],
  'output_schema':'New output has actual target_query_token_ids64/target_token_ids64 and cache_slot_mapping64; no bonus head is supplied. Keep original non-mathematical execution/profile fields from actual Native producer. The new verdict dispatch never feeds64heads to old5-query verifier. Fallback retains full old dict ABI.'},
 'protocol_and_KV':{
  'raw_conversion':[dict(raw_a=a,tree_a=min(a,3),delivered=min(a,3)+1,
    frontier='d4 (proved equal h3)' if a==4 else 'h'+str(a)) for a in range(5)],
  'packet':'Pad converted emitted row to original[N,5] with-1, accepted path range(tree_a) to[N,4] with-1. Keep raw_a0..4, raw_all4_match and actual target h0..h3 separately. Fourth comparison is real but supplies no fifth token.',
  'state_semantics':'Original record_verification receives proposed4/accepted<=3/delivered<=4. Full raw match has accepted path3 and frontier d4; full_primary_accept is false. EOS/remaining truncation, target_ended/windows and final delivery retain original logic.',
  'why_KV_valid':'Reject k0..3 needs past root plus accepted ancestors, at most4 computed slots. All4 raw match maps to path3: past root,d1,d2,d3 is materialized; next root d4 at P+4 is not yet evaluated. FastV5 four steps write exactly that same draft past. Never claim leaf d4 KV was evaluated.',
  'private_preflight':'Original Native5865 derives parent_count+1=5 and rejects mapping64. Private count4 lane must retain original dtype/device/capacity/unique/bounds/ticket/epoch checks plus exact current owner/query positions/source lease. Draft canonical deferred semantics stay original; target row mappings have4 actual slots. All other cases call captured original validator.',
  'compaction':'Accepted paths are only contiguous0..2, so original _tree_row_requires_kv_compaction naturally returnsFalse. Never call original compact_tree_round with logicalB4 plus4-slot mapping (its ABI requires5). Assert this invariant instead of forcing a compaction-skip flag.',
  'prestartup_capture':'Capture unmodified original_native_protocol_v75 before entry.install and keep engine._v75_original_protocol untouched for original components. Derive separate fresh private protocol200 with exact one-call dispatch difference and its own strict source contract; retain original finish/cleanup and original reducer observation. Do not mutate old compiled globals, spoof extracted-lanes-unchanged, late-reflect fresh_decode, or weaken _require_compiled.',
  'live_binding':'The live generated decode receives the same checked dispatch at the original self._preflight... call. Bind new lane context to the real current ticket/cohort before use and clear it in finally; state/next must not infer policy from shape alone.',
  'errors_and_lifetime':'New leader kernel/readback errors are carried in a validated shared status frame so all4 take the original fatal/cleanup path before commit; generic fallback also participates in the fixed framing required by the new integration. Never continue with stale KV. Keep actual indexed tensor/model/current-local-rank device equality; do not repeat193 unindexed engine.device equality error. Hold staging/device references through copy+replay+readback completion.'},
 'gain_calibration_and_conditionality':{
  'old_prior':'Existing26640 curve and old B4 online posterior describe BONUS output up to5; retain them only for old route. No truncation of old delivered totals or sharing (class,B4) accumulator with newpolicy.',
  'new_label':'B4-spine-four-comparisons-four-query-noBONUS-v200; units actual delivered tokens/owner-round before terminal-horizon clipping, max4. Keep raw acceptance/converted tree acceptance/delivered units distinct.',
  'fresh_samples':'At least the original36 independent class prefixes (12/class), recomputed by actual draft4 and target64 using fresh own histories; N16=12 scored+4 explicitly unscored filler. Record every raw head/verdict/delivery; do not truncate oldBONUS samples. Preserve own proof/calibration time outside formal full measurement.',
  'conservative_provider_proposal':'For prospective16normal/noE only, run the SAME original need/priority/whole-jump rules in a side-effect-free candidate namespace using separately calibrated new g(class,4); other n unchanged. Adopt that allocation only if its final actual grant is all B4 and exact lane is still valid. Otherwise use full original197 allocation with old g. Commit scheduler counters/state only for selected allocation, never double age/consume budget.',
  'uncertain_future_cohort':'Grant-time label is provisional; a changed verification cohort uses oldBONUS fallback and updates oldroute stats. Conservative new estimate may underpredict that fallback, but must never advertise old5token estimate as new4token output. Log provisional/actual policy and fallback reason. Exact cohort is rechecked, not predicted from future target output.',
  'explicit_negative_outcome':'If fresh new g4 has no positive jump over old g3 and original allocation consequently selects B3 or new lane never becomes eligible, preserve that zero-hit/negative result. Do not force B4, rename it B3, inflate g4, mix posteriors or claim an exercised optimization.',
  'reset_and_online':'Calibration prior is explicit/pinned; fixture/warm posterior is excluded. At each original generation reset, clear both route online tables. Production observations are keyed by actual route+class+B and exclude each route appropriate terminal horizon. Original acceptance EMA receives truthful converted counts; no fabricated full accept.',
  'requires_ROOT_choice_before_implementation':'This route-aware provider is the smallest honest closure of the post-grant predicate issue. If ROOT wants fixed original grants as an ablation, label that as an explicit frozen-policy diagnostic rather than calibrated dynamic g and obtain separate scope approval.'},
 'actual_qualification_before_full':[
  'Rerun current constructor201prefill/44target/114draft, original full527B4 and173M80, then197 current2/32 multiowner prefill and true nextB4; no inherited model receipt.',
  'New N16 G64/H64 cold-original/changed/restored actual F/T/F, natural flag transitions/extra-forward counters, actual entry identity, all64physical hidden and64head vs fresh independent sameM oracle, all64target layers K/V(query=128 tensors), protected past and exact metadata/currentHOST page checks.',
  'Three fresh complete new-policy traces with successive actual contexts/commits and own-next; two complete repetitions have zero original21 Native counters. Preserve actual input/proposal/position/owner/slot/context/table/epoch hashes and changed literal TTU signature/tasks where retained. No manual signatures/flags.',
  'Actual original noBONUS kernel and pure independently written integer reference cover mismatch0..3/all4, token0/duplicates/vocab, wrong dtype/shape/device, EOS/remaining, malformed ticket/owner/slot/context/epoch/alias negatives. Natural model accept counts remain actual, not forced; forced integer cases are carrier tests only.',
  'Real original controller tickets, publication once, original consensus, converted verdict/finish/cleanup and independent next after reject/all4 and oldBONUS-or-B5→new/new→fallback transitions. Exact queried-slot prefix validity and draft safe_before ledger are hard gates.',
  'Keep sameM math hard gates separate from crossM64vs80 head compatibility: retain actual64heads and old80 first-four heads from independently rebuilt identical prefixes, positions/diffs and accepted outputs; do not call crossM disagreement KV corruption or wholeAR equivalence. ROOT decides numerical-variant classification before any resulting formal claim.',
  'Current issuer only after all new gates and released fixture allocation. Original warm-to-fixedpoint, warm65/twozero65/single seal/full463 original arrival/output256/outerhostwall and callbacks remain; full must show new usage>0 and real generic B5/heterogeneous fallback>0 or report subset/no exercised-route claim. E fallback is retained but cannot be called device-covered if never reached.'
 ],
 'minimum_HOST_before_ready':[
  'Execute original fixed operator reference via explicit tensor carriers plus independently written scalar compare; verify every raw_a conversion and original state.apply_tree_verification/history/next positions. Do not mirror only the new implementation.',
  'Execute actual NativeMetadata producer and real NativePrefixCache with original host page validator for4query row positions crossing pages; wrongowner/missingpage/alias/epoch/device cases reject; no fifth slot allocated as evidence.',
  'Extract original preflight function body, demonstrate real count5/64 mismatch negative, then private corrected count4 physical validator over all16 actual tickets/cachemappings; real original finite/consensus/finish source remains exact.',
  'Actual prestartup capture→fresh namespace install→bounded+live dispatch bindings and original component capture identity; cold source-only installed namespace gate; exact inverse of each explicit AST change, no missing imports/globals or duplicate SCOPE kwargs.',
  'Actual route-aware original allocator with independently specified scalar cases: true all4 eligible, heterogeneous fallback, changed cohort fallback, no positive g4 jump/zero hit, selected-only counters/age, separate online/reset/horizon tables.',
  'New private guard real freeze/receipt constructor AST and baseline prepost/selflock/177 scoped .tmp storage/owned cleanup; inherited binaries verified in place, no environment clone/old large suite rerun.'
 ],
 'reporting_and_success_boundary':{
  'required_counts':['actual/cohort N and B histogram','provisional vs actual route and fallback reasons','actual64 vs admission80 queries','new/old g tables and sample counts','raw_a vs tree accepted vs delivered incl terminal truncation','actual new/fallback targetcalls and all4 status packets','draft4/head4/root_seed repair calls/query counts and known-KV holes','true Native key/validation/TTU/replay counters','whole original wall/463 completions/classTPOT/TTFT/GP/raw'],
  'no_inference':'Do not infer latency from20%query reduction, infer absence of repairs from raw_a, sum crossrole HOST phases, or transfer197 performance/AR/qualification. ROOT target>644.766raw is an experimental target, not a promised consequence.',
  'adoption':'adoptedFalse; whole-workload canonicalAR not established by these finite sameM gates. Job/full success and improvement are separately audited from mathematical current-component qualification.'},
 'source_pins':pins
}
path=D/'V200_B4_FOUR_QUERY_NO_BONUS_MINIMUM_IMPLEMENTATION_PLAN.json'
assert not path.exists()
path.write_text(json.dumps(out,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(path)
print(hashlib.sha256(path.read_bytes()).hexdigest(),path.stat().st_size)
