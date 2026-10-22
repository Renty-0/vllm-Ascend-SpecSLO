"""Proposed extra finite proof for nonempty refill; no installer or launcher.

Keep all original209 new-prefix/reference/root/next/Native-zero mathematics.
Supplement it with32 genuine old histories and hashes of every allocated old
KV page, including deferred/uncommitted slots, plus the actual old draft ledger.
This code has not run on an NPU and grants no serving permission.
"""
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import FunctionType

import prefill209 as original


def need(value,message):
    if value is not True:raise RuntimeError('HOST427 proposed NN proof: '+message)


def fixture(rows,n):
    need(n in (2,32) and len(rows)==65,'original finite prompt inventory')
    old=sorted(enumerate(rows),key=lambda v:(len(v[1]['token_ids']),v[0]))[:32]
    old_tokens=[list(row['token_ids'])for _,row in old]
    # Includes the original64 scratch reservation and the independent new
    # reference owners. No pool/active limit is increased for the oracle.
    reserve=lambda values:sum((len(p)+9+127)//128 for p in values)
    pages=64+2*reserve([r['token_ids']for r in rows[:n]])+reserve(old_tokens)
    need(pages<=544 and n+32<=64,'same actual544-page and64-active bounds')
    return dict(old_source_indices=[i for i,_ in old],old_tokens=old_tokens,
        old_ids=list(range(2*n,2*n+32)),planned_pages_with_scratch=pages,
        real_active_owners=n+32,independent_shadow_owners=n)


class Bank(original.Bank):
    def __init__(self,engine,rows,n):
        self.incumbent_fixture427=fixture(rows,n)
        super().__init__(engine,rows,n)
        self.incumbent_ids427=self.incumbent_fixture427['old_ids']
        self.incumbent_prompts427=self.incumbent_fixture427['old_tokens']
        # The original single incumbent is replaced by a genuine prompt from
        # the new declared finite fixture; the original209 case still runs
        # independently in the required base component.
        engine._release_cache_sequence(self.incumbent)
        for owner,p in zip(self.incumbent_ids427,self.incumbent_prompts427,strict=True):
            engine._activate_cache_sequence(owner,p,enable_prefix_caching=False)
            for pos in range(0,len(p)+9,128):engine._ensure_cache_capacity([owner],[pos])
        self.incumbent_tokens=list(self.incumbent_prompts427[0])
        self.tables=copy.deepcopy(engine.cache_allocation.block_tables)
        self.incumbent_contexts427=None

    def prepare_incumbents(self,native):
        e=self.engine
        states=[native.PearlPipelineState(list(p),len(p),max_tokens=256,ignore_eos=True)
                for p in self.incumbent_prompts427]
        roots=e._prefill_and_sample_target_batch(self.incumbent_prompts427,states,self.incumbent_ids427)
        need(len(roots)==32 and all(type(t)is int for t in roots),'genuine old target roots')
        contexts={owner:list(p)for owner,p in zip(self.incumbent_ids427,self.incumbent_prompts427,strict=True)}
        batches=[]
        if e.is_draft:
            for start in (0,16):
                ids=self.incumbent_ids427[start:start+16]
                local=states[start:start+16];current=roots[start:start+16]
                for state,root in zip(local,current,strict=True):
                    state.token_ids.append(root);state.committed_length+=1
                plans=[e._spec_rhythm_tree_plan(state,4)for state in local]
                value=e.draft_tree_forward(plans,current,ids,
                    committed_catchup_token_ids=[[t]for t in current],graph_padding_catchup_rows=[])
                need(value is not None and value['num_draft_tokens']==[4]*16,
                     'actual original old16/B4 draft, not a fabricated future ledger')
                for owner,root,row in zip(ids,current,value['draft_token_ids'],strict=True):
                    need(len(row)==8,'original full encoded primary and siblings')
                    contexts[owner].extend([root,*row[:3]])
                batches.append(dict(owners=ids,B=4,selected_candidates=64))
        self.incumbent_contexts427=contexts
        self.incumbent_prepared427=dict(actual_original_prefill_roots=True,old_roots=roots,
            original_old_draft_batches=batches,old_owner_count=32,real_active_owners=32+self.n)
        return self.incumbent_prepared427

    def snapshot_incumbents(self):
        import torch
        need(self.incumbent_contexts427 is not None,'actual old prefill/draft before protection')
        self.guard()
        pages=tuple(page for i in self.incumbent_ids427
                    for page in self.engine.cache_allocation.block_tables[i]if page>=0)
        need(len(pages)==len(set(pages)),'old owned physical pages are unique')
        slots=[p*128+j for p in pages for j in range(128)]
        values=original.query_KV(self.engine,torch.tensor(slots,dtype=torch.long,device=self.engine.device))
        need(len(values)==(56 if self.engine.is_draft else 128),'all role KV tensors')
        # Hash bytes even in uninitialized spare slots; this is an isolation
        # check, not a claim that unwritten spare slots contain finite values.
        physical=[original.tensor_hash(v)for v in values]
        known=getattr(self.engine,'_dynamic_tree_known_KV',{})
        ledger={i:copy.deepcopy(known[i])for i in self.incumbent_ids427}if self.engine.is_draft else None
        return dict(owners=self.incumbent_ids427,pages=list(pages),slots=len(slots),
            all_role_KV_pages_sha256=physical,actual_old_draft_ledger=ledger,
            valid_prefix=original.prefix_evidence(self.engine,self.incumbent_contexts427))

    def guard(self):
        result=super().guard()
        result.update(incumbent_owners=list(self.incumbent_ids427),
            planned_pages_with_scratch=self.incumbent_fixture427['planned_pages_with_scratch'],
            real_active_owners=32+self.n,independent_shadow_owners=self.n)
        return result


def project(function,changes,namespace):
    before=inspect.getsource(function);after=before
    for old,new,count in changes:
        need(after.count(old)==count,'exact original proof anchor '+old[:70])
        after=after.replace(old,new)
    reverse=after
    for old,new,count in reversed(changes):
        need(reverse.count(new)==count,'exact inverse proof anchor '+new[:70])
        reverse=reverse.replace(new,old)
    need(reverse==before,'entire original proof source inverse')
    exec(compile(after,__file__+'::proposed-original209-'+function.__name__,'exec',dont_inherit=True),namespace)
    return dict(original_sha256=hashlib.sha256(before.encode()).hexdigest(),
        proposed_sha256=hashlib.sha256(after.encode()).hexdigest(),entire_source_inverse=True)


def namespace():
    ns=dict(vars(original),Bank=Bank)
    seed="hidden=lane('incumbent-original-model',lambda:engine._run_packed_hidden(bank.incumbent_tokens,[bank.incumbent]*len(bank.incumbent_tokens),list(range(len(bank.incumbent_tokens))),use_aclgraph=False,use_fused_infer_attention=True).clone())"
    proof=project(original.run_case,(
        (seed,"prepared_incumbents=lane('incumbent32-real-prefill-and-old-draft',lambda:bank.prepare_incumbents(native))",1),
        ("require(bool(torch.isfinite(hidden).all()),'complete real incumbent hidden finite')",
         "require(prepared_incumbents['actual_original_prefill_roots']is True,'actual old model roots before protection')",1),
        ('prefix_evidence(engine,{bank.incumbent:bank.incumbent_tokens})','bank.snapshot_incumbents()',3),
        ("incumbent_prefix_protected=True,","incumbent_prefix_protected=True,all32_incumbent_allocated_pages_protected=True,incumbent_protection427=protected,incumbent_prepared427=prepared_incumbents,",1),
    ),ns)
    run=project(original.run,(
        ("'multiowner-eager-prefill-own-current-process'","'multiincumbent32-prefill427-own-current-process'",1),
        ('report=dict(report,audit=value)',
         'report=dict(report,audit=value,same_model_instance_id=id(engine.model),production_permission=False)',1),
        ('engine._v209_actual_prefill=report','engine._root427_incumbent_prefill_actual=report',1),
    ),ns)
    ns['audit']=audit
    return ns,dict(case=proof,run=run,NN_executed=False,production_permission=False)


def audit(directory):
    import gzip
    result=original.audit(directory)
    for rank in range(4):
        for pass_index in range(3):
            for n in (2,32):
                path=Path(directory)/f'rank{rank}-N{n}-pass{pass_index}.json.gz'
                row=json.loads(gzip.decompress(path.read_bytes()))
                lease=row['lease'];old=row['incumbent_protection427'];prepared=row['incumbent_prepared427']
                need(row['all32_incumbent_allocated_pages_protected']is True
                    and lease['incumbent_owners']==old['owners']==list(range(2*n,2*n+32))
                    and lease['real_active_owners']==32+n<=64
                    and lease['planned_pages_with_scratch']<=544,
                    'retained real old32 plus newN scope and unchanged resources')
                pages=[p for i in old['owners']for p in lease['block_tables'][i]if p>=0]
                need(old['pages']==pages and old['slots']==128*len(pages)
                    and len(old['all_role_KV_pages_sha256'])==(56 if rank==0 else 128),
                    'every actual old allocated KV slot protected in current model')
                need(prepared['actual_original_prefill_roots']is True and len(prepared['old_roots'])==32,
                    'genuine current old roots')
                if rank==0:
                    need(len(prepared['original_old_draft_batches'])==2
                        and all(r['B']==4 and r['selected_candidates']==64
                                for r in prepared['original_old_draft_batches'])
                        and set(map(int,old['actual_old_draft_ledger']))==set(old['owners']),
                        'actual old draft32 ledger protected after real original16+16 work')
    return dict(original209_new_prefix_math=result,passed=True,
        old_incumbents=32,new_N=[2,32],real_active_counts=[34,64],
        all_old_allocated_KV_pages_and_draft_ledger_protected=True,
        original_new_prefix_reference_root_next_and_twozero_gates_retained=True,
        Native_flags_assigned=False,production_permission=False,Raw=None,Goodput=None)
