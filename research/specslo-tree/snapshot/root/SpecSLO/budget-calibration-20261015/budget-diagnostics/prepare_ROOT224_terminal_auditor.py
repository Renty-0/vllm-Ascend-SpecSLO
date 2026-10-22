from pathlib import Path
import json,hashlib,ast
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';old=D/'audit_ROOT218_terminal_full_HOST_v3.py';P=B/'tree-FP32-tie-policy-full463-own-AR-v224';pins=json.loads((D/'ROOT_V224_SOURCE_PREPARED.json').read_text());s=old.read_text().replace('218','224').replace('tree-packed-KV-preflight-rank-aware-own-AR-full-v224',P.name)
s=s.replace("FREEZE='fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260'",'FREEZE='+repr(pins['freeze_sha256']))
s=s.replace("ARCHIVE='163960145a58f20f71241ce8f55c4db25abf3bc832fa44041ac1f1a7b547032d'",'ARCHIVE='+repr(pins['archive_sha256']))
s=s.replace('len(sources)==1874','len(sources)==1905').replace('source/archive1874','source/archive1905')
anchor="            result['actual_cached_dispatch']=cache_records(reports,need)";assert s.count(anchor)==1
extra='''            policies=[]
            for rank,report in enumerate(reports):
                policy=report['ROOT224_tie_policy'];initial=report['overlay']['ROOT224_target_tie_policy']
                need(policy['policy']=='FP32-refinement-of-complete-BF16-maximum-ties-cap4-v1'
                    and policy['rank']==rank and policy['installed']is(rank!=0)
                    and policy['installed_before_constructor']is True and policy['actual_engine_role_verified']is True
                    and policy['complete_ties_per_shard_cap']==4 and policy['overflow_policy']=='original minimum token ID',
                    'actual explicitly changed target tie policy and role')
                need(policy['model_weights_changed']is policy['model_body_changed']is policy['draft_sampler_changed']is False
                    and policy['greedy_tie_policy_changed']is True
                    and policy['weight_dtype']==policy['body_embedding_dtype']=='torch.bfloat16',
                    'original BF16 model/weights with declared new FP32 tie comparison')
                need(initial['python_head_entries']==0 and policy['python_head_entries']>0 if rank else initial['python_head_entries']==policy['python_head_entries']==0,
                    'new target wrapper actually entered during original fresh qualification')
                for name,digest in policy['files'].items():need(sha(name)==digest and freeze['files'].get(name)==digest,'new helper in exact frozen flat source map')
                need(sha(policy['original_source'])==policy['original_source_sha256'],'original native greedy source retained')
                reference=read(actual/'postfull-independent-FIA-AR'/f'rank{rank}-summary.json')
                need(reference['target_greedy_tie_policy']=='ROOT224 complete BF16 top ties up to4 per shard refined in FP32; original choice on overflow',
                    'independent own-AR explicitly declares same numerical rule')
                policies.append(policy)
            result['actual_explicit_new_target_tie_policy']=dict(by_rank=policies,old_BF16_tie_outputs_are_not_new_policy_reference=True,
                new_policy_applies_to_tree_and_independent_AR=True,model_weights_and_body_unchanged=True)
'''
s=s.replace(anchor,anchor+'\n'+extra)
s=s.replace('Same frozen209 math/protocol/layout with explicit cache, SLO admission1 and bounded N9to16 mixed-budget repair; budget grants can change. Source reviews do not transfer numerical qualification.',
 'Same frozen209 weights/body/protocol/layout and current budget, with an explicit new target FP32 maximum-tie rule applied to both tree and own-AR. This changes numerical selection among BF16 ties; original BF16 token results are not the new numerical reference. Fresh qualification is required.')
out=D/'audit_ROOT224_terminal_full_HOST_v1.py';assert not out.exists();ast.parse(s);out.write_text(s)
print(json.dumps(dict(path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),device_used=False,terminal_run_not_executed=True)))
