"""Prepared alternative: constant M80 FULL9; fresh model, no q1 capture."""
import ast,json
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');OLD=B/'tree-query1-FULL-numeric-isolation-diagnostic-v229';P=B/'tree-M80-FULL9-numeric-isolation-diagnostic-v230'
P.mkdir(exist_ok=False)
for p in OLD.iterdir():
    if p.suffix in('.py','.sh'):(P/p.name.replace('229','230')).write_text(p.read_text().replace(str(OLD),str(P)).replace('229','230'))
(P/'FIXTURE230.json').write_bytes((OLD/'FIXTURE229.json').read_bytes())
def sub(s,a,b):
    if s.count(a)!=1:raise RuntimeError('single source anchor '+a[:90])
    return s.replace(a,b)
p=P/'fresh230.py';s=p.read_text();s=sub(s,'fixed_M=True,split_queries=True','fixed_M=True,split_queries=False');s=s.replace('all_decode_attention_metadata_query1_FULL=True','all_decode_attention_metadata_M80_FULL9=True');p.write_text(s)
p=P/'metadata230.py';s=p.read_text();s=sub(s,'    from canonical230 import canonical\n    return inputs,pos,canonical(meta)','    return inputs,pos,meta');p.write_text(s)
p=P/'history230.py';s=p.read_text().replace('all_decode_attention_metadata_query1_FULL=True','all_decode_attention_metadata_M80_FULL9=True');p.write_text(s)
p=P/'body_cost230.py';s=p.read_text().replace('q1_FULL1','q5_FULL9').replace('existing q1','existing q5 FULL9').replace('existing-q1-cost','existing-q5-cost');p.write_text(s)
p=P/'guard230.py';s=p.read_text();s=sub(s,"r['shape']['S']==80 and r['shape']['q']==list(range(1,81))","r['shape']['S']==16 and len(r['shape']['q'])==16 and r['shape']['q'][-1]==80 and r['shape']['mask_shape']==[16,1,9,4096]")
s=s.replace('isolated_existing_q1_body_cost_only','isolated_existing_q5_FULL9_body_cost_only').replace('canonical M80 one-query FULL segments','canonical M80 FULL9 segments').replace('existing q1 body replay cost','existing q5 FULL9 body replay cost');p.write_text(s)
p=P/'check230_CPU.py';s=p.read_text().replace('tuple(meta.tree_attention_mask.shape)==(80,1,1,4096)','tuple(meta.tree_attention_mask.shape)==(16,1,9,4096)').replace("mask=meta.tree_attention_mask[5*j+k,0,0]","mask=meta.tree_attention_mask[j,0,k]")
s=sub(s,"meta.actual_seq_lengths_q==tuple(range(1,81))and meta.sequence_lens==tuple(p+1 for p in frame['positions'])","meta.actual_seq_lengths_q==tuple(range(5,81,5))and meta.sequence_lens==tuple(frame['sequence_lens'])")
s=sub(s,"m.tree_attention and v['inputs'].numel()==80 and m.actual_seq_lengths_q==tuple(range(1,81))","m.tree_attention and v['inputs'].numel()==80 and len(m.actual_seq_lengths_q)==16 and m.actual_seq_lengths_q[-1]==80")
s=sub(s,"(~m.tree_attention_mask[i,0,0,n:]).any()","(~m.tree_attention_mask[i,0,:,n:]).any()")
s=s.replace('exact_per_query_FULL_causal_and_branch_masks_verified=True','exact_FULL9_causal_and_branch_masks_verified=True')
p.write_text(s)
(P/'ROOT_SOURCE_REVIEW230.json').write_text(json.dumps(dict(status='source_only_not_scheduled',purpose='Separate original FULL9 constant M80 from q1 canonicalization; fresh model before all captures',
    main_path='M80 FULL9 with original exact request-level KV lengths and q partitions',eager_ablations='same six M/Q controls; q1 eager only, never captured in this process',
    absolute_cost='already qualified same FULL9 graph only, zero new captures',original_sampler_unchanged=True,production_installed=False,neural_qualified=False,performance_qualified=False,adopted=False),indent=2)+'\n')
for p in P.glob('*.py'):ast.parse(p.read_text())
print(json.dumps(dict(directory=str(P),source_prepared=True,NPU_started=False)))
