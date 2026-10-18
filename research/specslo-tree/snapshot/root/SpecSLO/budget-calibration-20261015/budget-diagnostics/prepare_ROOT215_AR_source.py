"""Source-only candidate214 plus a separate post-timing independent FIA AR check."""
import ast,difflib,hashlib,json
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015')
DONOR=B/'tree-cost-aware-mixed16-budget-full-v214'
P=B/'tree-cost-aware-mixed16-with-own-FIA-AR-full-v215'
assert not P.exists();P.mkdir()
records=[]
for src in sorted(DONOR.iterdir()):
    if src.suffix not in ('.py','.sh'):continue
    dst=P/src.name.replace('214','215')
    dst.write_text(src.read_text().replace(str(DONOR),str(P)).replace('214','215'))
    records.append(dict(source=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),new=str(dst)))
runtime=P/'runtime215.py';s=runtime.read_text()
old='("     report[\'Native_final\']=", "     report[\'ROOT215_cached_parent_dispatch\']=extension.cache_snapshot()\\n     report[\'Native_final\']=",1)'
# Keep the original full timing and final controls untouched; reference runs later.
assert old in s
new='("     report[\'Native_final\']=", "     report[\'postfull_independent_FIA_AR\']=extension.postfull_AR(engine,native,stage,report[\'windows\'],output_directory)\\n     report[\'ROOT215_cached_parent_dispatch\']=extension.cache_snapshot()\\n     report[\'Native_final\']=",1)'
runtime.write_text(s.replace(old,new))
ext=P/'extension215.py';s=ext.read_text()
s+='''

def postfull_AR(engine,native,stage,windows,output):
    from reference215 import run
    return run(engine,native,stage,windows,output)
'''
ext.write_text(s)

P.joinpath('reference215.py').write_text('''"""Independent target-only AR, same FIA backend; never teacher force candidate tokens.

Q80/S16 consumes only the first query output of each q5 causal segment.
Four future inputs are canaries, not proposals. Every next real token and KV
prefix comes from this reference's own prompt and previous AR outputs.
The full candidate performance clock has already ended before this code runs.
"""
import gzip,hashlib,json
from pathlib import Path


def need(value,message):
    if value is not True:raise RuntimeError('ROOT215 AR: '+message)


def digest(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def frame(prompts, outputs, indices, *, future=0):
    need(0<len(indices)<=16 and len(set(indices))==len(indices),'one nonempty unique <=16 owner subset')
    need(all(0<=i<len(prompts) and outputs[i] for i in indices),'actual own independent frontiers')
    tokens=[];ids=[];positions=[];lengths=[]
    for i in indices:
        position=len(prompts[i])+len(outputs[i])-1
        need(position+4<4096,'original FIA context capacity')
        tokens.extend([outputs[i][-1]]+[future]*4);ids.extend([i]*5)
        positions.extend(range(position,position+5));lengths.append(position+5)
    for i in range(16-len(indices)):
        tokens.extend([0]+[future]*4);ids.extend([463+i]*5);positions.extend(range(5));lengths.append(5)
    need(len(tokens)==len(ids)==len(positions)==80 and len(lengths)==16,'exact original80/S16 physical geometry')
    return dict(tokens=tokens,ids=ids,positions=positions,sequence_lens=lengths,
                actual_seq_lengths_q=list(range(5,81,5)),real_owners=list(indices),
                real_head_rows=list(range(0,5*len(indices),5)))


def compare(candidate,reference):
    need(candidate.keys()==reference.keys(),'all exact request IDs, no drops')
    rows=[]
    for rid,actual in reference.items():
        expected=candidate[rid]
        need(len(actual)==len(expected)==256,'complete256 own AR outputs')
        mismatch=next((i for i,(a,b)in enumerate(zip(actual,expected))if a!=b),None)
        rows.append(dict(request_id=rid,equal=mismatch is None,first_mismatch_zero_based=mismatch,
            candidate_token=None if mismatch is None else expected[mismatch],
            reference_token=None if mismatch is None else actual[mismatch]))
    return dict(requests=len(rows),equal_requests=sum(r['equal']for r in rows),
        mismatching_requests=sum(not r['equal']for r in rows),rows=rows,
        exact_all_tokens_equal=all(r['equal']for r in rows))


def retained_trace(windows):
    trace=windows['traces'][-1]['original_trace']
    need(trace['role']=='full463' and trace['scope_count']==463,'completed original full463 required first')
    artifact=Path(trace['artifact'])
    with gzip.open(artifact,'rt')as f:value=json.load(f)
    need(value['rank']in range(4) and value['scope_count']==463,'own actual rank raw trace')
    return artifact,value


def target_step(engine, data):
    import torch
    from mapper173 import GREEDY
    runner=engine.graph_runner;entry=runner.target_entries[GREEDY]
    need(runner.graph_cache_sealed is True and entry.runtime_validated is True
        and GREEDY not in runner.disabled_entry_keys,'existing sealed naturally qualified80 graph')
    positions,meta=engine._prepare_attention_metadata(data['ids'],data['positions'],True)
    need(meta.use_fused_infer_attention is True and not getattr(meta,'tree_attention',False)
        and list(meta.actual_seq_lengths_q)==data['actual_seq_lengths_q']
        and list(meta.sequence_lens)==data['sequence_lens']
        and tuple(meta.block_tables.shape)==tuple(meta.request_block_tables.shape)==(16,32)
        and tuple(meta.attention_mask.shape)==(2048,2048),'original causal FIA80 exact metadata ABI')
    inputs=torch.tensor(data['tokens'],dtype=torch.long,device=engine.device)
    output=runner.run_target_greedy([inputs],[positions],[meta],engine.draft_vocab_size)[0]
    values=output.cpu().tolist()
    need(len(values)==80 and all(type(v)is int and 0<=v<engine.draft_vocab_size for v in values)
        and runner.target_entries[GREEDY]is entry,'complete original80 greedy head and same graph object')
    return dict(heads80=values,real_tokens=[values[i]for i in data['real_head_rows']],
        metadata=dict(positions=positions.cpu().tolist(),slots=meta.slot_mapping.cpu().tolist(),
            contexts=meta.context_lens.cpu().tolist(),block_tables=meta.request_block_tables.cpu().tolist(),
            actual_seq_lengths_q=list(meta.actual_seq_lengths_q),sequence_lens=list(meta.sequence_lens)))


def run(engine,native,stage,windows,output):
    import torch.distributed as dist
    from scope_v146 import NATIVE_COUNTERS,FAILURES
    output=Path(output);folder=output/'postfull-independent-FIA-AR'
    stage('AR215/create-private-reference-output',lambda:folder.mkdir(exist_ok=False)if engine.rank==engine.topology.draft_leader_rank else None)
    artifact,trace=stage('AR215/own-completed-trace',lambda:retained_trace(windows))
    prompts=trace['tokenized_inputs'];requests=trace['requests']
    candidate={r['request_id']:r['completion_token_ids']for r in trace['generate_results']}
    groups=[r['sequence_ids']for r in trace['standalone_prefill_calls']]
    need(sorted(i for group in groups for i in group)==list(range(463))
         and all(0<len(group)<=32 for group in groups),'same original prefill groups and order, all463 once')
    need(engine.cache_allocation is None and engine.cache_block_tables is None
         and engine.graph_runner.graph_cache_sealed is True,'fresh own cache only after full completion/seal')
    reference={};cohort_summaries=[];start={k:engine.graph_metrics()[k]for k in NATIVE_COUNTERS}
    for ordinal,owners in enumerate(groups):
        lane=lambda name,fn:stage('AR215/cohort'+str(ordinal)+'/'+name,fn)
        source=[list(prompts[i])for i in owners]
        local_ids=list(range(len(source)))
        lane('fresh-independent-prompt-allocation',lambda:engine._allocate_cache(source,enable_prefix_caching=False,reserve_sequence_capacity=True))
        states=[native.PearlPipelineState(list(p),len(p),max_tokens=256,ignore_eos=True)for p in source]
        roots=lane('original-same-cohort-independent-prefill',lambda:engine._prefill_and_sample_target_batch(source,states,local_ids))
        actual=[[r]for r in roots];steps=[];canaries=[]
        for begin in range(0,len(local_ids),16):
            selected=local_ids[begin:begin+16]
            for step in range(255):
                data=frame(source,actual,selected)
                row=lane('AR'+str(begin)+'-step'+str(step),lambda:None if engine.is_draft else target_step(engine,data))
                if step in (0,127,254):
                    alternate=frame(source,actual,selected,future=151643)
                    other=lane('future-canary'+str(begin)+'-'+str(step),lambda:None if engine.is_draft else target_step(engine,alternate))
                    if not engine.is_draft:
                        need(row['real_tokens']==other['real_tokens'],'future dummy inputs must not change first causal query outputs')
                        canaries.append(dict(segment_begin=begin,step=step,real_owners=selected,
                            zeros=row['real_tokens'],alternate=other['real_tokens'],equal=True))
                if not engine.is_draft:
                    for i,token in zip(selected,row['real_tokens'],strict=True):actual[i].append(token)
                    steps.append(dict(segment_begin=begin,step=step,heads80=row['heads80'],
                        metadata=row['metadata']if step in(0,127,254)else None))
        def finish():
            need(engine.cache_allocation is not None,'own reference allocation retained')
            if not engine.is_draft:
                need(all(len(v)==256 for v in actual),'all reference tokens generated by own AR history')
                for i,v in zip(owners,actual,strict=True):reference[requests[i]['request_id']]=v
            packet=dict(rank=engine.rank,reference_sha256=None if engine.is_draft else digest(actual))
            gathered=[None]*4;dist.all_gather_object(gathered,packet,group=engine.groups.verification_coordination_group)
            need({r['rank']for r in gathered}==set(range(4)) and len({r['reference_sha256']for r in gathered if r['rank']!=0})==1,'all3 target ranks agree independently before any output sharing')
            tables=engine.cache_allocation.block_tables;pages=[v for t in tables for v in t if v>=0]
            need(len(pages)==len(set(pages))<=544 and len(tables)==527,'independent reference unique physical pages in original pool')
            return dict(ordinal=ordinal,original_owner_indices=owners,request_ids=[requests[i]['request_id']for i in owners],
                first_roots=roots,reference_completions=None if engine.is_draft else actual,
                reference_generated_on_this_rank=not engine.is_draft,candidate_tokens_ever_used_as_inputs=False,
                own_prompt_only_allocation=True,own_prefix_KV_only=True,steps=steps,canaries=canaries,
                rank_agreement=gathered,actual_unique_pages=len(pages),block_tables=tables)
        row=lane('retainable-independent-cohort',finish)
        path=folder/f'rank{engine.rank}-cohort{ordinal}.json.gz'
        lane('save-independent-cohort',lambda:path.write_bytes(gzip.compress(json.dumps(row,separators=(',',':')).encode(),mtime=0)))
        cohort_summaries.append(dict(ordinal=ordinal,owners=len(owners),steps=0 if engine.is_draft else len(steps),
            exact_reference_sha256=None if engine.is_draft else digest(actual),actual_unique_pages=row['actual_unique_pages']))
        lane('release-own-reference-cache',engine._release_cache)
    after={k:engine.graph_metrics()[k]for k in NATIVE_COUNTERS};delta={k:after[k]-start[k]for k in start}
    need(not any(delta.values()) and all(delta[k]==0 for k in FAILURES),'reference replays only already sealed graph paths; original21 all zero')
    result=dict(schema='ROOT215.postfull-independent-FIA80-AR.v1',rank=engine.rank,
        completed=True,backend='original target TP3 BF16 causal FIA Q80/S16; only first query per owner consumed',
        source_artifact=str(artifact),source_artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        same_prefill_cohorts=True,reference_trajectories_generated=0 if engine.is_draft else len(reference),
        reference_only_prompt_plus_own_outputs=True,candidate_KV_or_tokens_seeded=False,
        performance_clock_already_finished=True,cohorts=cohort_summaries,
        comparison=None if engine.is_draft else compare(candidate,reference),
        Native=dict(before=start,after=after,delta=delta),cache_released=engine.cache_allocation is None,
        cross_backend_PA_or_official_AR_equivalence_claimed=False,formal_Goodput=False,adopted=False)
    path=folder/f'rank{engine.rank}-summary.json'
    stage('AR215/save-independent-summary',lambda:path.write_text(json.dumps(result,indent=2)+'\\n'))
    return result
''')

# Host checks cover the independent recurrence and empty/missing/mismatch cases.
host=P/'host215.py';s=host.read_text()
tests='''

class ReferenceTests(unittest.TestCase):
    def test_reference_frame_only_own_frontier_and_causal_layout(self):
        from reference215 import frame
        prompts=[[10,11,12],[20]*130];outputs=[[40,41],[50]]
        x=frame(prompts,outputs,[1,0]);self.assertEqual(x['tokens'][:10],[50,0,0,0,0,41,0,0,0,0])
        self.assertEqual(x['positions'][:10],list(range(130,135))+list(range(4,9)))
        self.assertEqual(x['ids'][:10],[1]*5+[0]*5)
        self.assertEqual(x['sequence_lens'][:2],[135,9]);self.assertEqual(x['real_head_rows'],[0,5])
        self.assertEqual(x['tokens'][10:],[0]*70);self.assertEqual(x['ids'][10:15],[463]*5)
        alt=frame(prompts,outputs,[1,0],future=151643)
        self.assertEqual(alt['tokens'][::5],x['tokens'][::5])
        self.assertEqual(prompts,[[10,11,12],[20]*130]);self.assertEqual(outputs,[[40,41],[50]])
        with self.assertRaises(RuntimeError):frame(prompts,outputs,[])
        with self.assertRaises(RuntimeError):frame(prompts,outputs,[0,0])
        with self.assertRaises(RuntimeError):frame([[0]*4092],[[1,2]],[0])

    def test_complete_exact_comparison_never_turns_mismatch_to_success(self):
        from reference215 import compare
        a={'a':[1]*256,'b':[2]*256};b={'a':[1]*256,'b':[2]*255+[3]}
        r=compare(a,b);self.assertFalse(r['exact_all_tokens_equal']);self.assertEqual(r['equal_requests'],1)
        self.assertEqual(r['rows'][1]['first_mismatch_zero_based'],255)
        with self.assertRaises(RuntimeError):compare(a,{'a':[1]*256})
        with self.assertRaises(RuntimeError):compare({'a':[1]*255},{'a':[1]*255})
'''
s=s.replace("if __name__=='__main__':",tests+"\n\nif __name__=='__main__':")
host.write_text(s)
for r in records:
    old=Path(r['source']).read_text().replace(str(DONOR),str(P)).replace('214','215')
    new=Path(r['new']).read_text()
    r['new_sha256']=hashlib.sha256(new.encode()).hexdigest()
    r['patch_after_namespace_projection']=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True)))
    if Path(r['new']).suffix=='.py':ast.parse(new)
ast.parse((P/'reference215.py').read_text())
(P/'PROJECTION215.json').write_text(json.dumps(dict(schema='ROOT215.postfull-AR-source-only.v1',
    donor=str(DONOR),donor_freeze_sha256=hashlib.sha256((DONOR/'FREEZE214.json').read_bytes()).hexdigest(),
    projection=records,new_reference_module_sha256=hashlib.sha256((P/'reference215.py').read_bytes()).hexdigest(),
    performance_code_delta_from214=False,reference_runs_after_original_full_clock=True,
    NPU_started=False,source_peer_review=False),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='source_prepared_unfrozen',directory=str(P),NPU_started=False)))
