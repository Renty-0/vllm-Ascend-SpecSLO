"""Uninstalled CPU layout for contiguous per-query branch KV tails.

Share complete committed-prefix pages, copy only a partial last prefix page
and visible current ancestors to unused dedicated scratch pages. No NPU use.
"""
def layout(plan,table,scratch_pages,*,block_size=128):
    physical=plan.cache_positions.tolist();logical=plan.positions.tolist();mask=plan.attention_mask
    prefix=physical[0]
    if prefix!=logical[0]or block_size<=0 or any(p<0 for p in table):raise ValueError('valid original root and source pages required')
    if len(set(scratch_pages))!=len(scratch_pages)or set(scratch_pages)&set(table):raise ValueError('independent unique scratch pages required')
    complete=prefix//block_size;used=0;rows=[];copies=[]
    for row,(p,l)in enumerate(zip(physical,logical,strict=True)):
        visible=(~mask[row]).nonzero().flatten().tolist()
        if visible[:prefix]!=list(range(prefix))or len(visible)!=l+1 or visible[-1]!=p:raise ValueError('exact complete prefix and ordered own ancestral path required')
        if visible==list(range(l+1)):
            rows.append(dict(tables=list(table),length=l+1,original_physical_query=p,copy_pages=0));continue
        tail=visible[complete*block_size:];count=(len(tail)+block_size-1)//block_size
        pages=scratch_pages[used:used+count]
        if len(pages)!=count:raise ValueError('bounded dedicated scratch pages exhausted')
        used+=count;new=list(table)
        for j,page in enumerate(pages):new[complete+j]=page
        for j,source in enumerate(tail):
            src=table[source//block_size]*block_size+source%block_size
            dst=pages[j//block_size]*block_size+j%block_size
            copies.append((src,dst))
        rows.append(dict(tables=new,length=l+1,original_physical_query=p,copy_pages=count))
    return dict(rows=rows,copy_slots=copies,scratch_pages=used,committed_prefix=prefix)

def main():
    import importlib.util,json,sys
    from pathlib import Path
    import torch
    torch.set_num_threads(1)
    native=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/tree.py')
    spec=importlib.util.spec_from_file_location('_ROOT_branch_tail_original_tree',native);tree=importlib.util.module_from_spec(spec);sys.modules[spec.name]=tree;spec.loader.exec_module(tree)
    cases=[];table=list(range(100,132));scratch=list(range(1000,1064))
    for prefix in(0,1,124,125,126,127,128,129,252,255,511,1023,4000):
        raw=tree.cached_cpu_tree_speculation_plan(2,4,prefix,4096,candidate_budget=8)
        selections=[list(range(b))for b in range(1,9)]+[[0,1,2,3,4+a]for a in range(4)]
        for selected in selections:
            plan=tree.pack_selected_tree_plan(raw,selected);out=layout(plan,table,scratch);copied={dst:src for src,dst in out['copy_slots']}
            assert len(copied)==len(out['copy_slots'])
            for q,row in enumerate(out['rows']):
                visible=(~plan.attention_mask[q]).nonzero().flatten().tolist()
                expected=[table[p//128]*128+p%128 for p in visible]
                actual=[copied.get(row['tables'][p//128]*128+p%128,row['tables'][p//128]*128+p%128)for p in range(row['length'])]
                assert actual==expected and row['length']==int(plan.positions[q])+1
                assert row['tables'][:prefix//128]==table[:prefix//128]
                if len(selected)<=4:assert row['copy_pages']==0
            cases.append(dict(prefix=prefix,selected=selected,scratch_pages=out['scratch_pages'],copied_KV_positions=len(copied)))
    bad=tree.pack_selected_tree_plan(tree.cached_cpu_tree_speculation_plan(2,4,127,4096,candidate_budget=8),[0,1,2,3,7])
    negatives=[]
    for name,pages in [('empty',[]),('duplicate',[1000,1000]),('source_alias',[100,1000])]:
        try:layout(bad,table,pages)
        except ValueError:negatives.append(name)
        else:raise AssertionError('invalid scratch accepted '+name)
    capacities=[]
    for N in range(1,17):
        for branches in range(N+1):
            for Q in range(6*branches+2*(N-branches),min(80,6*branches+5*(N-branches))+1):
                extra=80-Q
                if extra==1:continue # old dummy-plan ABI cannot represent a root-only padding row.
                max_dummy=min(32,extra//2)
                assert extra==0 or max_dummy>=1 and 9*max_dummy>=extra
                assert 64-max_dummy>=2*branches
                capacities.append((N,branches,Q,max_dummy,2*branches))
    assert 'torch_npu'not in sys.modules
    result=dict(status='HOST_layout_oracle_passed_uninstalled',original_tree_cases=len(cases),negative_cases=negatives,capacity_geometries=len(capacities),
        max_branch_scratch_pages=max(x[4]for x in capacities),existing_64_scratch_rows_suffice_if_unused_pages_can_be_borrowed=True,
        source_order_and_all_visible_KV_preserved=True,only_partial_prefix_and_current_ancestry_copied=True,
        caveats=['No NPU implementation or graph capture exists for this layout.','Copy work would be needed at every target layer; latency unknown.','M80 admission and root-only padding remain unimplemented.','Runtime must independently lease unused scratch pages and prove no aliases.'],
        details=cases,NPU_used=False,production_installed=False,neural_qualified=False,performance_qualified=False)
    out=Path(__file__).with_name('ROOT_QUERY1_BRANCH_TAIL_LAYOUT_CPU_RESULT.json')
    with out.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k!='details'}))
if __name__=='__main__':main()
