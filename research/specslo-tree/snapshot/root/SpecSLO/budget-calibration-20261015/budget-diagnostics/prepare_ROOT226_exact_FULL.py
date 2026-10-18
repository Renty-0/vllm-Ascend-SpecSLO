"""Repair ROOT225's comparator without changing any frozen input."""
import ast, datetime, hashlib, json
from pathlib import Path

B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
OLD=B/'tree-canonical-FULL-attention-numeric-cost-diagnostic-v225'
P=B/'tree-exact-FULL-attention-matched-reference-diagnostic-v226'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with p.open('x')as f:json.dump(v,f,indent=2);f.write('\n')
def sub(s,a,b):
    if s.count(a)!=1:raise RuntimeError('single source anchor '+a[:100])
    return s.replace(a,b)

P.mkdir(exist_ok=False)
for p in OLD.iterdir():
    if p.suffix not in ('.py','.sh')and p.name!='FIXTURE225.json':continue
    s=p.read_text().replace(str(OLD),str(P)).replace('225','226')
    (P/p.name.replace('225','226')).write_text(s)

p=P/'history226.py';s=p.read_text()
s=s.replace('bucketed_eager_reference','exact_eager_reference').replace('current_real16_bucketed_eager_graph_exact','current_real16_exact_metadata_original_matcher_passed')
s=sub(s,'                from vllm_ascend.spec_decode.pearl.native_graph import bucket_tree_fia_attention_metadata\n','')
s=sub(s,'                meta=bucket_tree_fia_attention_metadata(meta,block_size=engine.config.kvcache_block_size)\n','')
s=sub(s,"                need(torch.equal(hidden,final[0]),'all80 real current FULL hidden vectors equal original bucketed eager reference')", "                matched=engine.graph_runner._target_outputs_match((hidden,),(final[0],))\n                need(matched,'all80 current exact-length FULL hidden pass original Native matcher')")
s=sub(s,"                need(all(torch.equal(a,b)for a,b in zip(graph_KV,eager_KV,strict=True)),'all128 query KV tensors match original bucketed eager reference')\n                return True", "                return dict(original_matcher_passed=matched,all80_hidden_bitwise_equal=torch.equal(hidden,final[0]),\n                    all80_hidden_max_abs=float((hidden.float()-final[0].float()).abs().max()),\n                    query_KV_bitwise_equal=all(torch.equal(a,b)for a,b in zip(graph_KV,eager_KV,strict=True)),\n                    identical_exact_metadata_used=True,bucket_helper_used=False)")
s=sub(s,"            step('exact-current-real16-bucketed-eager-model-reference',exact_eager_reference)","            eager_match=step('exact-current-real16-same-metadata-eager-model-reference',exact_eager_reference)")
s=sub(s,'current_real16_exact_metadata_original_matcher_passed=True)','current_real16_exact_metadata_original_matcher_passed=eager_match[\'original_matcher_passed\'],same_metadata_eager_graph_observation=eager_match)')
s=sub(s,"        results.append(dict(case=case['name'],request_id=case['request_id'],neural_math_on_this_rank=target,detail=detail,", "        results.append(dict(case=case['name'],request_id=case['request_id'],neural_math_on_this_rank=target,detail=detail,")
s=sub(s,"    return dict(schema='ROOT226history.incremental-common-history-numeric.v1'", "        from runtime226 import save\n        save(output,f'rank{engine.rank}-history-complete-case{len(results)}.json',results[-1])\n    return dict(schema='ROOT226history.incremental-common-history-numeric.v1'")
p.write_text(s)
p=P/'fresh226.py';s=p.read_text();s=sub(s,"    return dict(schema='ROOT226fresh.shared-prefix-numeric-diagnostics.v1'", "        from runtime226 import save\n        save(output,f'rank{engine.rank}-fresh-complete-case{len(results)}.json',results[-1])\n    return dict(schema='ROOT226fresh.shared-prefix-numeric-diagnostics.v1'");p.write_text(s)
p=P/'probes226.py';s=p.read_text();s=sub(s,"    history=run_history", "    from runtime226 import save\n    save(output,f'rank{engine.rank}-fresh-complete.json',fresh)\n    history=run_history");p.write_text(s)
p=P/'guard226.py';s=p.read_text().replace('current_real16_bucketed_eager_graph_exact','current_real16_exact_metadata_original_matcher_passed').replace('all80 real-row current canonical FULL model matcher','all80 real-row exact-length FULL original Native model matcher');p.write_text(s)
p=P/'check226_CPU.py';s=p.read_text();start=s.index(" node=copy.deepcopy(next(n for n in ast.parse((N/'native_graph.py')")
end=s.index(" sys.path[:0]",start);s=s[:start]+s[end:]
start=s.index("  frame=case['reference_frames'][-1]");end=s.index('  e._release_cache()',start);s=s[:start]+s[end:]
s=s.replace('original_bucket_logic_used=True,adjacent_context_bucket_checks=bucket_checks,','actual_runtime_uses_exact_KV_lengths=True,bucket_helper_not_used=True,')
p.write_text(s)

receipt=B/'tree-causal80-small-cohort-role-aware-audit-full-v209/receipt-canonical-FULL-root-v225-four-contexts-body-cost-v1.json'
r=json.loads(receipt.read_text());assert r['status']=='failed'and r['sources_unchanged']and r['owned_remaining']==[]and r['devices_after']['all_idle']
failure=dict(schema='ROOT225.invalid-comparator-terminal-correction.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),receipt=str(receipt),receipt_sha256=digest(receipt),
    status='failed_invalid_comparison_setup',root_cause='diagnostic eager-only bucket helper compared against actual exact-length FULL graph',
    runtime_source_exact_length=True,bucket_helper_not_called_by_actual_FULL_path=True,failed_evidence_preserved=True,
    current_FULL_cpu_updates_not_skipped_by_bucket=True,performance_qualification=False,adopted=False,
    pre_verifier=r['pre_verifier']['actual_result'],post_verifier=r['post_verifier']['actual_result'],owned_remaining=[],all_four_idle=True,
    correction='Retire the earlier claim that current FULL buckets adjacent contexts or thereby skips attention task updates.')
save(D/'ROOT_V225_TERMINAL_INVALID_COMPARATOR_AND_CORRECTION.json',failure)
for p in P.glob('*.py'):ast.parse(p.read_text())
save(P/'ROOT_SOURCE_REVIEW226.json',dict(status='source_prepared_HOST_CPU_pending',no_frozen_files_edited=True,
    repaired_comparator='identical exact metadata; original unmodified Native matcher; bitwise hidden/KV separately reported',
    completed_cases_saved_before_later_phases=True,original_sampler_unchanged=True,neural_qualified=False,performance_qualified=False,adopted=False))
print(json.dumps(dict(directory=str(P),prepared=True,device_started=False)))
