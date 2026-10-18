"""Bounded canonical one-query FULL attention hypothesis; no production install."""
import ast,json
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015')
OLD=B/'tree-exact-FULL-attention-matched-reference-diagnostic-v226'
P=B/'tree-query1-FULL-canonical-numeric-cost-diagnostic-v227'
P.mkdir(exist_ok=False)
for p in OLD.iterdir():
    if p.suffix not in('.py','.sh'):continue
    s=p.read_text().replace(str(OLD),str(P)).replace('226','227')
    (P/p.name.replace('226','227')).write_text(s)
# All recorded token IDs, positions, lengths and input hashes remain identical.
(P/'FIXTURE227.json').write_bytes((OLD/'FIXTURE226.json').read_bytes())
def sub(s,a,b):
    if s.count(a)!=1:raise RuntimeError('single source anchor '+a[:80])
    return s.replace(a,b)
p=P/'metadata227.py';s=p.read_text();s=sub(s,'    return inputs,pos,meta','    from canonical227 import canonical\n    return inputs,pos,canonical(meta)');p.write_text(s)
p=P/'fresh227.py';s=p.read_text()
s=sub(s,"    plans=[]\n", "    frame=dict(frame,M=80,S=16) # same physical body M for all bounded fresh variants\n    plans=[]\n")
s=sub(s,'    meta=padded(meta,9)','    meta=padded(meta,9)\n    from canonical227 import canonical\n    meta=canonical(meta)')
s=sub(s,"len(meta.actual_seq_lengths_q)==frame['S']","len(meta.actual_seq_lengths_q)==frame['M']")
s=s.replace('all_decode_attention_metadata_forced_FULL9=True','all_decode_attention_metadata_query1_FULL=True')
p.write_text(s)
p=P/'history227.py';s=p.read_text().replace('all_decode_attention_metadata_forced_FULL9=True','all_decode_attention_metadata_query1_FULL=True');p.write_text(s)
p=P/'guard227.py';s=p.read_text().replace('four contexts','four contexts, canonical M80 one-query FULL segments')
s=sub(s,"require(all(r['shape']['tree']is True for r in fresh)","require(all(r['shape']['tree']is True and r['shape']['M']==80 and r['shape']['S']==80 and r['shape']['q']==list(range(1,81)) for r in fresh)")
p.write_text(s)
(P/'canonical227.py').write_text('''"""Preserve each query's original FULL visibility while splitting FIA segments."""
from dataclasses import replace
def canonical(meta):
    import torch
    if not meta.tree_attention or not meta.use_fused_infer_attention:raise ValueError('original FULL metadata required')
    if meta.context_lens.device.type!='cpu':raise ValueError('original host context lengths required')
    ends=meta.actual_seq_lengths_q;start=0;owners=[];rows=[]
    mask=meta.tree_attention_mask;extent=int(mask.shape[2])
    if len(ends)!=int(mask.shape[0])or meta.request_block_tables.shape[0]!=len(ends):raise ValueError('original request alignment')
    for owner,end in enumerate(ends):
        count=end-start
        if not 0<count<=extent:raise ValueError('original query partition fits mask')
        owners.extend([owner]*count);rows.extend(owner*extent+k for k in range(count));start=end
    M=start;lengths=tuple(int(n)for n in meta.context_lens.tolist())
    if len(lengths)!=M or any(not 0<n<=int(mask.shape[3])for n in lengths):raise ValueError('exact per-query host KV lengths')
    device=mask.device
    oi=torch.tensor(owners,dtype=torch.long,device=device);ri=torch.tensor(rows,dtype=torch.long,device=device)
    tables=meta.request_block_tables.index_select(0,oi)
    qmask=mask.reshape(-1,int(mask.shape[3])).index_select(0,ri).reshape(M,1,1,int(mask.shape[3]))
    return replace(meta,block_tables=tables,request_block_tables=tables,
        actual_seq_lengths_q=tuple(range(1,M+1)),sequence_lens=lengths,tree_attention_mask=qmask)
''')
# CPU fixtures validate the new ABI independently of execution guards.
p=P/'check227_CPU.py';s=p.read_text()
s=s.replace("tuple(meta.tree_attention_mask.shape)==(16,1,9,4096)","tuple(meta.tree_attention_mask.shape)==(80,1,1,4096)")
s=s.replace("'all ordinary q5 frames packed as FULL9'","'all ordinary q5 frames split as FULL query1'")
s=s.replace("mask=meta.tree_attention_mask[j,0,k]","mask=meta.tree_attention_mask[5*j+k,0,0]")
s=sub(s,"    need(bool(meta.tree_attention_mask[:,:,5:,:].all()),'unused query rows fully masked')", "    need(meta.actual_seq_lengths_q==tuple(range(1,81))and meta.sequence_lens==tuple(p+1 for p in frame['positions']),'exact one-query partition and own visible extent')")
s=sub(s,"   v=prepare_frame(e,case,frame);need(v['metadata'].tree_attention,'both formerly branched and causal fixtures force FULL9');count+=1", "   v=prepare_frame(e,case,frame);m=v['metadata'];need(m.tree_attention and v['inputs'].numel()==80 and m.actual_seq_lengths_q==tuple(range(1,81)),'canonical M80 query1 FULL');\n   for i,n in enumerate(m.sequence_lens):need(not bool((~m.tree_attention_mask[i,0,0,n:]).any()),'no visible KV truncated by per-query physical extent')\n   count+=1")
s=s.replace('full_causal_masks_and_padding_verified=True','exact_per_query_FULL_causal_and_branch_masks_verified=True')
p.write_text(s)
for p in P.glob('*.py'):ast.parse(p.read_text())
(P/'ROOT_SOURCE_REVIEW227.json').write_text(json.dumps(dict(status='source_only_unqualified',hypothesis='one-query FULL segments remove future-window-dependent reductions; M80 body for all fresh variants',
    original_visibility_preserved=True,own_physical_KV_lengths=True,original_sampler_unchanged=True,original_Native_graph_validation_retained=True,
    caveat='Branched queries still contain masked physical KV holes; no full AR equivalence assumed.',production_installed=False,performance_qualified=False,adopted=False),indent=2)+'\n')
print(json.dumps(dict(directory=str(P),prepared=True,NPU_started=False)))
