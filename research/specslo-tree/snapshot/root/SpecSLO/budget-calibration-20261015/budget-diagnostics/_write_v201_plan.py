from pathlib import Path
import hashlib,json,datetime

B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-empty-active-cohort-causal80-active32-full-v197'
O=B/'tree-high-normal-cap4-native-families-v148'
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
proposal=B/'tree-normal-b4-final-kv5-cohort16-full-v201'
assert not proposal.exists()
paths=[P/'FROZEN_SOURCE197.json',P/'runtime197.py',P/'extension197.py',P/'scope197.py',P/'component197.py',
 P/'empty197.py',P/'prefill197.py',P/'next4_197.py',P/'trace197.py',P/'audit197.py',
 O/'fast_tree_draft_v5.py',O/'root_seed.py',O/'chain_metadata.py',O/'tail_hidden_observer.py',
 O/'draft_observer_v146.py',O/'draft_v146.py',O/'kv_ledger.py',O/'backbone_selection.py',
 O/'finite_inventory_plan.py',O/'startup_receipts_v75.py',O/'original_tree_cycle_v75.py',
 O/'original_native_protocol_v75.py',O/'runtime_tree_decode.py',N/'native_graph.py',N/'native_engine.py']
pins={str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in paths}
out={
 'schema':'V201.normal-B4-final-KV-only-five-step-minimum-source-plan.v1',
 'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'status':'source_only_plan_not_implementation_authorized',
 'candidate_directory_proposed':str(proposal),'candidate_directory_created':False,
 'scope':dict(Torch_or_model_or_NPU_imported=False,device_access=False,old_suites_run=False,
  current197_actual_or_receipt_read=False,frozen_sources_modified=False),
 'sole_production_parent':dict(path=str(P),freeze_sha256=pins[str(P/'FROZEN_SOURCE197.json')]['sha256'],
  retained='empty-active-only cohort admission, active32/config64/queued463/storage527/cache544, cap4 and lowN trueB5, C96/E96/global192; original463x256/arrival/TTFT/TPOT/outerwall unchanged'),
 'recommended_first_arm':{
  'domain':'Only current normal work with N16, all actual logical B4 on the fixed primary spine, no eager_parent_sources/E work, greedy mode and four prediction stages. Every owner has no actual missing committed suffix according to original FastV5.missing_committed_suffix and the authoritative known-KV ledger.',
  'operation':'Reserve all P..P+4 positions, build five original PA metadata steps at physical rows16, run original run_draft_greedy(...final_kv_only=True,return_confidence=True). The first four steps predict d1..d4 with original top2/FP32 confidence; step5 consumes d4 only, writes its actual KV, produces no logits/head/output column.',
  'fallback':'Any missing committed suffix, other N, heterogeneous B, B5/fork, E, stochastic/unsupported or absent current issuer takes the exact original197/root_seed method before model work. Never catch a model/Native validation failure and fallback after a partial graph.',
  'not_in_initial_arm':'No root_seed tail4 variant, no new target mapper/key/protocol, no change to g/allocator/selection, no final fifth proposal or head, no B4parent eager capability, no expansion beyond N16.'},
 'why_this_differs_from_V200':{
  'V200_problem':'At identical first3 draft tokens and four target heads, four-query noBONUS B4 delivers exactly oldB3+BONUS. Honest steady-state gain has no fourth-candidate delivery advantage; do not manufacture a positive B4 marginal from12-sample noise.',
  'V200_valid_limited_use':'An explicit frozen-original-grant cost ablation may still measure less target work/changed repair, but is not a new g-optimal dynamic policy and does not establish the whole-positive-jump allocator will choose it. No such candidate is created by this note.',
  'V201_preserves':'Original B4 target evaluates root+d1+d2+d3+d4 (Q80 at N16), original BONUS verifier may deliver d1..d4+h4 (max5), and all four actual proposal/head/confidence values retain their original meaning. g prior/online units and original need/priority/grants therefore need no verification-policy replacement, provided current first4 encoded outputs are proven equal.',
  'tradeoff':'Pay one additional draft transformer/PA stage for every selected owner, including rejected/terminal rows, to possibly avoid a later deferred d4 repair and its packed-root submission. Not a free step or an extra delivered token. No GP prediction.'},
 'source_contract_confirmed':{
  'native_API':'native_graph.py1691-1703 accepts final_kv_only;1804-1812 encodes final-kv separately into the exact draft_entries key;NativeDraftACLGraphEntry.final_kv_only is a capture-time field, not a flag to mutate.',
  'actual_top2':'fast_tree_draft_v5.py35-50 and installed tail_hidden_observer.observed both call model on every metadata step, then continue before compute_logits/softmax/topk on the final KV-only step. Five metadata steps return[N16,4,2] packed top2 values, not5 proposal columns.',
  'private_anchors':'Do not set proposal_steps=5 everywhere: separate proposal_steps4/model_steps5. Keep row primary4+secondary4/raw8, original score8 and mean8 denominator. Query/model-stage accounting uses5N=80; generated candidate count remains8N=128.',
  'metadata':'chain_metadata.build accepts1..8 steps and preserves original current-allocation table gather, per-step position/slot/context and PA NativeMetadata ABI. Its caller must ensure every queried page before its initial gather.',
  'extra_page':'Original Fast.reserve_query_pages only loops plan.depth4, so it does NOT reserve P+4. A private helper must explicitly ensure all5 positions for all16 owners before metadata build, validate current owner/page/generation/no alias/capacity, and allow original allocation failures to abort.',
  'ledger':'Original Fast ledger writes[root]+predicted[:-1] (4 queries). New lane must write[root]+all4 primary tokens (5 actual queries) at P..P+4. safe_before staysP; do not advance a committed boundary toP+5 or claim extra tokens were accepted. Existing record_overwrite semantics remove descendant claims after corrections.'},
 'new_Native_domain':{
  'exact_key':['draft-greedy:151936|steps:5|paged|final-kv|confidence',16],
  'inventory_delta':'Exactly one new original runner.draft_entries key on draft rank; no new entries, target_entries, target G/H keys or kernel/profile. Original explicit chain keys are steps1..4 without final-kv, and original28 generic regression keys belong to target_entries. Current object/flags must be observed; no copied entries or manually assigned validation.',
  'execution':'Retain graph_lane=None/stable_task_barrier=False and original PA/replay-first/event/workspace/finite handling. No generic Native entries or target FIA mode used for the new draft graph.',
  'parallel_PA_support':'native_graph.py4282-4296 explicitly accepts step_count4+int(final_kv_only)=5 with output.shape[1]==4. It still requires the original enabled two-thread executor, correct PA task classes, per-task unique ExternalEvents, aligned lengths/workspaces/streams and no incompatible mode. Availability is not a claimed current success.',
  'expected_tasks_if_original_collector_contract_holds':'Draft has28 layers:5x28=140 PA attention tasks versus112 for ordinary4step. Record actual tasks_per_step/tasks/update counters rather than inventing140 if the collector differs.',
  'critical_validation_gap':'Native capture/runtime output matching intentionally passes attention_metadatas[:-1] when final_kv_only, so runtime_validated/validated_real_row_count closes only the four output columns/first4 slot layouts. It does NOT independently prove final d4 hidden/KV or the fifth slot mapping. A separate current all5-step sameM/56KV hard gate is mandatory.'},
 'transition_and_repair_closure':{
  'fresh_prefill_to_new':'Original197 fully materializes each prompt, publishes its sampled root exactly once, then original planner makes normal draft work. Preserve original Fast initialization of a previously unseen owner from this proven prefill contract; do not fabricate a prompt ledger or copy candidate KV into reference.',
  'new_all4_accept':'Candidate computed root,d1,d2,d3,d4 at P..P+4. Original target accepts4 and publishes BONUS h4 as next root atP+5. Required committed past through d4 is already real draft KV, so original missing_committed_suffix should returnempty on the next same-lane call. This must be observed, not inferred from a fullaccept boolean.',
  'new_reject':'If k<4 accepted, next root is correction h_k atP+k+1. Original known ledger may retain speculative descendants, but the next real root write plus record_overwrite/replaced written ledger must invalidate descendants with changed ancestry. Equal token IDs alone cannot certify stale contextual KV.',
  'old_to_new':'OldBONUS/B5 may leave a real missing committed suffix. For the initial arm, any owner with repair causes the entire N16 group to use original root_seed+tail3. Nothing is truncated to force the new graph. Record mixed old/new-ledger causes and eligibility fallback counts.',
  'new_to_old':'Original route receives a truthful ledger with extra d4 KV. missing_committed_suffix remains authoritative; it may skip a genuinely unnecessary repair and otherwise executes original repair. B5/fork/heterogeneous outputs, original target compaction and correction remain unchanged.',
  'coverage_limitation':'No-missing-only may exercise fewer frames than static N16/allB4 counts. After an old repair fallback, a subsequent allaccept can leave another old-style hole; there is no guaranteed migration into the new lane. Report genuine new hits and old repair fallbacks; zero/low hits is a result, not permission to mark unknown suffixes valid.',
  'no_E':'Even though four primary draft nodes are physically written, this new proof does not authorize B4 rolling-parent continuation, promotion or slot moves. Existing P3/eager filter stays unchanged.'},
 'root_seed_tail4_alternative_not_recommended_first':{
  'real_layout':'Original packed root+repair PA computes root and first head. Tail4 at P+1..P+4 with final_kv_only=True consumes d1,d2,d3,d4, predicts only d2,d3,d4 (three top2 columns). Concatenate seed1+tail3 to original4 pairs. Ledger includes[root]+d1..d4. Total model stages5 plus any repair tokens, still four heads.',
  'second_new_key':['draft-greedy:151936|steps:4|paged|final-kv|confidence',16],
  'parallel_caveat':'Its step_count4/output3 does NOT match the original gamma4-full-chain PA parallel gate (expected5/output4), so original updater rejects that fast subpath and uses its serial updater; this is supported fallback, not the same two-thread scope.',
  'extra_obligations':'Need independent original root PA+explicit root head+tail4/finalKV oracle, actual repairs0/1/multi and physical packed buckets/metadata plus new final slot/ledger/next checks, an additional current Native entry and differentiated costs. It is a separate scope expansion, not a one-line root_seed3→4 replacement.',
  'decision':'Start with one-key no-missing lane to isolate finalKV work and preserve complete old repair fallback. Consider this repair-aware second lane only if its new-key/HOST/numerical scope is explicitly approved and real coverage/cost justifies it.'},
 'minimum_implementation_boundary_if_authorized':{
  'private_draft201':'Install an outer checked draft dispatch AFTER original197 setup; retain exact original root_seed method as fallback. Derive private Fast function/body in fresh globals with explicit5metadata/finalTrue/5written/count anchors. Keep canonical Fast/root_seed/tail observer modules and Native _execute_draft/graph methods intact; original installed execute_top2 already handlesfinalKV.',
  'no_protocol_patch':'Original prestartup protocol capture remains before entry.install. Keep original target M80/S16 issuer, physical5query mapping, original five-head verifier packet, count5 preflight, compaction/finish/state apply/cleanup/publication and target_ended unchanged. No V200 count4 adaptation or universal new verifier packet is needed.',
  'qualification_lane':'Only after the current original197 component proof, permit explicit bounded private qualification token for the new graph in fresh own banks. Production issuer is issued only after sameM/protocol/next/twozero qualification releases fixture allocation. Bind engine/runner/newentry/currentbaseproof/actual method identity, not a bool or previous receipt.',
  'source_IO':'Thin fresh197 namespace/runtime/trace/scope/check/guard projection, preserving all empty-active admission and private IO paths. Read-only parent archives verified in place; no clone377/environment/binary archive. Expose actual new entry key in source/loaded-origin checks.',
  'controls':'Preserve original allocator method identity, cap4/lowN5/priority/need/g source, original target mapper and exact target lifecycle. If a source adapter changes a canonical module path, pin that real path and prove cold installednamespace readiness before model startup rather than relying on source retention alone.'},
 'minimum_current_math_and_protocol_actor':[
  'Rerun original constructor201/44/114, full527 B4+M80 and197 genuine multiowner prefill+normal-next component in the same process. Keep new dispatch disabled during those exact baseline components; do not inherit197 device qualification.',
  'New draft N16 five-step graph actual cold/changed/restored F/T/F with original Native flag transitions, actual entry object and validated real-row count16, exact five metadata fields/current owners/slots/contexts/page tables, protected past hashes and allphysical finite.',
  'Independent fresh reference owners write their own full prefix, then execute original four PA model+top2 stages atM16 and ONE original PA model call consuming their own predicted d4 atP+4, with no fifth head. It must not call new candidate helper or copy any candidate KV. Compare first4 fullphysical hidden and all[N16,4,2] encoded/score8/mean8 exactly; compare fifth fullphysical hidden and every layer K/V at all5 queried positions with original matcher, including d4 explicitly.',
  'Show actual five model evaluations and four head evaluations in the real captured/eager producer contract; original tail observer retains five nonaliasing hidden tensors. Native output matcher excludes fifth metadata, so do not substitute its marker for this gate. Retain query/slot/d4 KV evidence and finite sticky flags.',
  'Three fresh complete bounded traces with real original targets and tickets/commit, both natural accepts/rejections as they occur, own next after every cycle, changed contexts and actual Native TTU lengths/tasks. The second and third complete traces give two original21zero deltas once keys are resident; do not rename a cold passzero.',
  'Qualify new→new, new→original, and oldBONUS/B5→original repair fallback→new if naturally reachable. Independent reference must use actual committed tokens/corrections and its own ledger. Cover page edgeP mod128=123/124, reject descendants, owner reuse/allocation generation change, terminal remaining/EOS and all4 failure/cleanup. Forced integer verdict cases are carrier-only, never naturalaccept claims.',
  'Bounded common-input cost pairs use fresh independent histories and include complete draft/repair/target/original commit/ownnext scopes with actual role endpoints. Record stage5 paid on rejected rows and subsequent root_seed avoidance only where real counters support it; do not sum crossrole HOST times or extrapolate GP.',
  'Issue current production permission only after all hard gates, then unchanged fixedpoint prewarm/warm65/twozero65/one seal/full463 original clocks. All unsupported B5/heterogeneous/otherN/E routes remain original and require truthful current fallback coverage; no new route hits means no exercised optimization claim.'
 ],
 'minimum_HOST_only_before_source_ready':[
  'Execute actual Fast.execute_top2 function AST with explicit model/head tensor carriers and true fifth metadata: five model calls, four compute_logits/softmax/topk calls, four output pairs, final input equals real fourth primary. Independently assert no head5 using operation counters rather than only output shape.',
  'Execute actual chain_metadata producer and NativeMetadata dataclass with real NativePrefixCache boundary carriers at P+4 crossing128; oldreserve4 negative must expose missing page while newensure5 yields distinct exact slots. Do not infer5metadata from advertisedsteps.',
  'Execute actual ledger/missing_committed_suffix/record_overwrite functions through allacceptBONUS/rejection/correction/oldhole transitions. Old4written negative must retain missingd4; new5written nohole only when the model really wrote it. Wrongancestor/same-token descendants invalidate.',
  'Execute original key formula and original parallelPA predicate AST with fiveposition/output4 case and root_seed fourposition/output3 rejection; confirm finalkv marker lives in new captured entry and no old flag/object is mutated.',
  'Cold actual private installation/readiness, original prestartup capture and parent197 callbacks/guard freeze+receipt constructor exact AST. Preserve original parent HOST results without rerunningoldlarge suites. HOST carriers do not confer device mathematical/taskupdate qualification.'
 ],
 'required_reporting':{
  'strategy_label':'B4-BONUS-unchanged-target-normal-finalKV5-noMissing-N16-v201',
  'counts':['normalN16/allB4/noE prospective frames','eligible noMissing frames and fallback reason perowner','actual fifth-step hits and actual new draft key executions','model stages5/head stages4/rawnodes8/query5N and pagegrowth','raw accepted0..4/visible BONUS/truncation with old definitions','next missing suffix and original root_seed repair-query/submission counts by originating route when actually retained','newentry natural validation/changed TTU/tasks/skip/parallelaccept-or-reject counters','sameM first4 encoded equality and final d4 hidden/56KV result','full463 originalwall/raw/GP/classTPOT/TTFT without source-only forecasts'],
  'g':'Keep original B4 BONUS g prior and actual production online update unchanged. No new g-policy label is mixed intoold because verification/proposal topology isunchanged; still record strategy separately for analysis. If actual first4 output equivalence fails, stop rather than claiming identical mathematical output/g.',
  'risk':'Initial five-step costs more even when no later repair is avoided; actual domain may be small; more PA updates and draftgraph time may consume overlap window. Underlying source support and preserved target semantics do not prove gain or canonical whole463 equivalence.',
  'qualification_and_adoption':'No current math/device/GP result exists forV201; adoptedFalse. All new flags/keys/current proof, full completion, performance and eventual adoption remain separately audited.'},
 'source_pins':pins
}
path=D/'V201_NORMAL_B4_FINAL_KV_ONLY_FIVE_STEP_MINIMUM_SOURCE_PLAN.json'
assert not path.exists();path.write_text(json.dumps(out,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(path);print(hashlib.sha256(path.read_bytes()).hexdigest(),path.stat().st_size)
