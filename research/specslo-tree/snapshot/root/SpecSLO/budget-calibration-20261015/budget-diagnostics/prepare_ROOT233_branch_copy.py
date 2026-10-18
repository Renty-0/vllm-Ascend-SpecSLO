"""Fresh bounded diagnostic for contiguous branch KV tails; frozen232 untouched."""
import ast,hashlib,json
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');OLD=B/'tree-canonical-M80-independent-branch-math-v232';CORE=B/'tree-canonical-M80-FULL9-full463-own-AR-v231';P=B/'tree-contiguous-branch-tail-M80-math-v233'
P.mkdir(exist_ok=False)
for p in OLD.iterdir():
    if p.suffix not in('.py','.sh'):continue
    name=p.name.replace('232','233');s=p.read_text().replace(str(OLD),str(P)).replace('232','233')
    s=s.replace('M80-FULL9-independent-branch-math','M80-contiguous-branch-independent-math')
    (P/name).write_text(s)
for name in('branch_tail231.py','check_branch_tail231_CPU.py'):(P/name).write_bytes((CORE/name).read_bytes())
for name in('HOST_MAPPER231_RESULT.json','CPU_REFERENCE_SCHEDULE231_RESULT.json','CPU_BANK_PADDING231_RESULT.json','CPU_BRANCH_TAIL231_RESULT.json'):(P/name).write_bytes((CORE/name).read_bytes())
(P/'FIXTURE233.json').write_bytes((OLD/'FIXTURE232.json').read_bytes())
def sub(s,a,b):
    if s.count(a)!=1:raise RuntimeError('single source anchor '+a[:80])
    return s.replace(a,b)
p=P/'install233.py';s=p.read_text();a=s.index('def activate(');s=s[:a]+'''def activate(engine,native):
    from branch_tail231 import install
    return install(engine,native)
''';p.write_text(s)
p=P/'probes233.py';s=p.read_text().replace('activate(native)','activate(engine,native)')
s=sub(s,"    result=math(engine,native,stage,bank,output=output)","    from runtime233 import save\n    save(output,f'rank{engine.rank}-independent-bank.json',dict(rank=engine.rank,bank=bank['evidence'],roots=bank['roots']))\n    result=math(engine,native,stage,bank,output=output)")
s=s.replace('weights_or_model_body_changed=False','weights_changed=False,branch_attention_cache_view_modified=True').replace('decode_metadata_forced_FULL9=True','linear_FULL9_and_branch_copy_FULL10=True');p.write_text(s)
p=P/'neural_oracle231.py';s=p.read_text()
a='    return result\n\ndef prepare';extra='''    for anchor in range(4):
        budgets=[5]*12+[2,1,1];m,s,pad=envelope(15,79,'full9')
        result.append(dict(name=f'N15-fork-padding79-anchor{anchor}',N=15,budgets=budgets,anchor=anchor,M=m,S=s,padding=pad,plan_family='full9',query_count=79,fixture_budget_reduction=None))
    return result

def prepare'''
s=sub(s,a,extra)
a="    inputs,pos,meta=make_host_tree_attention_metadata(engine,pp,rr,dd,sequence_ids=ss,allow_causal_fast_path=False);meta=padded(meta,9)\n    need(tuple(meta.tree_attention_mask.shape)==(case['S'],1,9,4096)and meta.actual_seq_lengths_q[-1]==80,'same actual FULL metadata ABI')\n    hidden=engine.model(inputs,pos,meta).clone()"
b="    state=engine._branch_tail_state\n    inputs,pos,meta,enabled=state.last\n    baseline=state.last_record.get('baseline_original_layout',False)\n    from branch_tail231 import is_fork\n    S=case['S']+(sum(is_fork(p)for p in pp)if enabled else 0);Q=10 if enabled else 9\n    need(tuple(meta.tree_attention_mask.shape)==(S,1,Q,4096)and meta.actual_seq_lengths_q[-1]==80,'actual original-order canonical FULL metadata ABI')\n    hidden=state.eager_last()"
s=sub(s,a,b)
s=sub(s,"return dict(tokens=graph_tokens,KV=graph_KV,hidden=hidden[:case['query_count']].clone(),", "return dict(branch_layout=state.last_record,attention_segments=S,mask_query_capacity=Q,tokens=graph_tokens,KV=graph_KV,hidden=hidden[:case['query_count']].clone(),")
s=sub(s,'target=not engine.is_draft;records=[]','target=not engine.is_draft;records=[];negative_controls=[]')
s=sub(s,"        actual=lane('actual-production-target'", "        if case['name']=='N1-fork5-anchor0':\n            from baseline233 import run as negative\n            negative_controls.append(negative(engine,case,bank,plans,rows,lane,output))\n        actual=lane('actual-production-target'")
a="            matcher=engine.graph_runner._target_outputs_match\n            need(torch.equal(actual['tokens'],own['tokens'])"
b="            matcher=engine.graph_runner._target_outputs_match\n            from baseline233 import measurement,retain\n            metrics=measurement(engine,case,actual,own)\n            if not(metrics['tokens_equal']and metrics['hidden_original_matcher_passed']and metrics['KV_original_matcher_passed']):retain(output,f'rank{engine.rank}-FAILED-'+case['name']+'.json',metrics)\n            need(torch.equal(actual['tokens'],own['tokens'])"
s=sub(s,a,b)
s=sub(s,"return dict(case=case,real_predictions=case['query_count']", "return dict(branch_layout=actual['branch_layout'],attention_segments=actual['attention_segments'],mask_query_capacity=actual['mask_query_capacity'],case=case,real_predictions=case['query_count']")
s=sub(s,"return dict(rank=engine.rank,cases=records,", "return dict(rank=engine.rank,old_layout_negative_controls=negative_controls,cases=records,")
p.write_text(s)
for name in('runtime233.py','guard233.py'):
 p=P/name;s=p.read_text().replace('151','155');p.write_text(s)
p=P/'ROOT_SOURCE_REVIEW233.json';p.write_text(json.dumps(dict(status='source_prepared_device_unrun',cases=155,
    branch_KV_copy='constant1792 unique copy destinations, original BF16 cache; contiguous logical prefix+ancestry per side query',
    graph_contract='linear original FULL9; side queries split into their own FULL segment, mask query capacity10 distinguishes copy graph from no-copy graph',
    reference='independently prefilled own spine/side KV, q1 eager reference; copy state disabled for reference',
    negative_control='original layout on first failed N1-fork5-anchor0 retained separately; never counted as a passed case',
    original_sampler_unchanged=True,original_Native_matcher_unchanged=True,no_production_qualification=True,no_performance_claim=True,adopted=False),indent=2)+'\n')
for p in P.glob('*.py'):ast.parse(p.read_text())
print(json.dumps(dict(directory=str(P),source_prepared=True,NPU_started=False)))
