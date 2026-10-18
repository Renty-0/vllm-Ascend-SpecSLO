"""Keep frozen unrun227; avoid repeating known unqualified generic causal cost."""
import ast,json
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015')
OLD=B/'tree-query1-FULL-canonical-numeric-cost-diagnostic-v227'
P=B/'tree-query1-FULL-vs-FULL9-numeric-cost-diagnostic-v228'
P.mkdir(exist_ok=False)
for p in OLD.iterdir():
    if p.suffix not in('.py','.sh'):continue
    (P/p.name.replace('227','228')).write_text(p.read_text().replace(str(OLD),str(P)).replace('227','228'))
(P/'FIXTURE228.json').write_bytes((OLD/'FIXTURE227.json').read_bytes())
def sub(s,a,b):
    if s.count(a)!=1:raise RuntimeError('single source anchor '+a[:80])
    return s.replace(a,b)
p=P/'metadata228.py';s=p.read_text()
start=s.index('    if not full:');end=s.index('    from vllm_ascend',start)
s=s[:start]+"    if not full:raise ValueError('ROOT228 diagnostic is explicitly FULL only')\n"+s[end:]
reference=(B/'tree-exact-FULL-attention-matched-reference-diagnostic-v226/metadata226.py').read_text()
reference=reference[reference.index('def pack('):].replace('def pack(','def q5_full_reference(',1)
start=reference.index('    if not full:');end=reference.index('    from vllm_ascend',start)
reference=reference[:start]+"    if not full:raise ValueError('ROOT228 original FULL9 reference only')\n"+reference[end:]
p.write_text(s+'\n'+reference)
p=P/'body_cost228.py';s=p.read_text().replace('from metadata228 import pack','from metadata228 import pack,q5_full_reference').replace("('causal','FULL9')","('q5_FULL9','q1_FULL1')").replace("('causal','FULL9','FULL9','causal')","('q5_FULL9','q1_FULL1','q1_FULL1','q5_FULL9')")
s=sub(s,"            pair=[pack(engine,f,full=name=='FULL9')for f in(frame,shifted)]", "            fn=q5_full_reference if name=='q5_FULL9'else pack\n            pair=[fn(engine,f,full=True)for f in(frame,shifted)]")
s=sub(s,'includes_original_metadata_copy_replay_and_attention_updates=True,model_head_excluded=True,','includes_original_metadata_copy_replay_and_attention_updates=True,model_head_excluded=True,\n        comparison=\'M80 original FULL9 q5 versus canonical FULL1 q1; both exact KV lengths\',')
p.write_text(s)
p=P/'history228.py';s=p.read_text();s=sub(s,"            if mode=='reference'and case['original_owner']==34:","            if mode=='reference'and case['original_owner']==34:\n                from runtime228 import save\n                save(output,f'rank{engine.rank}-history-case2-reference-complete-before-cost.json',dict(observation=observation,formal_Goodput=False))")
p.write_text(s)
for p in P.glob('*.py'):ast.parse(p.read_text())
(P/'ROOT_SOURCE_REVIEW228.json').write_text(json.dumps(dict(status='source_only_unqualified',hypothesis='M80 and one-query FULL exact lengths',
    original227_frozen_unrun=True,reason_for_new_namespace='ROOT226 generic causal cost graph failed original validation; do not repeat it',
    body_cost_comparison='original q5 FULL9 versus canonical q1 FULL1, exact lengths, M80 both, ABBA',original_Native_validation_retained=True,
    same_original_four_fixtures=True,original_sampler_unchanged=True,production_installed=False,performance_qualified=False,adopted=False),indent=2)+'\n')
print(json.dumps(dict(directory=str(P),prepared=True,NPU_started=False)))
