"""Actual Native HOST layouts, exact grid formula and private capture-key test."""
import ast,copy,inspect,json,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_graph.py')
def main():
    sys.path[:0]=[str(P),str(B/'tree-alternate-chain-target-host-v269'),str(B/'tree-prefix-grid-byte-KV-copy-v278')]
    import check269_HOST as prior
    source=Path(prior.__file__).read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='main')
    stop=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='results'for t in n.targets))
    ns=dict(vars(prior),P=P,B=B)
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(node.body[:stop]),type_ignores=[])),str(P/'check_grid296_HOST.py')+'::original269-HOST-setup','exec'),ns)
    engine=ns['engine'];state=ns['state'];native=ns['native'];pad=ns['pad'];layout=ns['layout']
    from prefix278 import select_grid
    from oracle296 import cases,prepare
    from audit_static_calls296 import expected_grid,audit_calls
    fixture=json.loads((P/'FIXTURE296.json').read_text());prefixes=fixture['prefixes']
    reports=[];calls=[];seen=set()
    from capture_key296 import install,key_for,projected
    import tree_target_replay as replay
    text=N.read_text();original=next(n for n in ast.walk(ast.parse(text))if isinstance(n,ast.FunctionDef)and n.name=='_run_target')
    native_source='\n'.join(text.splitlines()[original.lineno-1:original.end_lineno])+'\n'
    generated=replay.patched_source(native_source)
    namespace=dict(torch=ns['torch'],_dynamic_tree_replay_submit=replay.submit)
    exec(compile(generated,str(Path(replay.__file__))+':_run_target','exec',dont_inherit=True),namespace)
    class Runner:pass
    runner=Runner();runner._run_target=types.MethodType(namespace['_run_target'],runner)
    meta=object();mockstate=types.SimpleNamespace(static_current={'physical':(None,None,meta),'copy_enabled':True},enabled=True,copy280_bucket=128)
    mock=types.SimpleNamespace(graph_runner=runner,_branch_tail_state=mockstate)
    proof=install(mock);assert proof['original_body_AST_inverse']is True
    base_key=('target-greedy:151936|steps:1|tree-fia-full:buffers:unchanged',(120,))
    keys=[]
    for grid in(128,256,512,1024,1792):
        mockstate.copy280_bucket=grid;key=key_for(mock,runner,[meta],base_key)
        assert key[1]==base_key[1]and key[0]==base_key[0]+'|ROOT296:copy-grid:'+str(grid)
        keys.append(key)
    assert len(set(keys))==5
    mockstate.static_current['copy_enabled']=False;mockstate.enabled=False
    assert key_for(mock,runner,[meta],base_key)==base_key
    for case in cases():
        plans,rows=prepare(case,prefixes);n=case['N']
        pp,rr,dd,ss,real=pad(engine,plans,list(range(41,41+n)),rows,list(range(n)),[],[])
        native.NativePearlEngine.target_tree_forward(engine,pp,rr,dd,sequence_ids=ss,return_logits=False,real_tree_count=real)
        desc=layout(pp,ss,engine.cache_allocation.block_tables,engine._tree_graph_scratch_sequence_ids)
        grid=select_grid(desc['source'],desc['dest'],rows=69632)if desc['enabled']else 0
        assert grid==expected_grid(case),(case['name'],grid,expected_grid(case))
        record=dict(state.static_graph_calls[-1],copy_grid=grid,
            actual_capture_key=repr((base_key[0]+('|ROOT296:copy-grid:'+str(grid)if grid else''),base_key[1])))
        family=(11 if grid else 10,grid);cold=family not in seen
        for mode in(['capture_replay','replay','replay']if cold else['replay']):
            calls.append(dict(record,execution_mode=mode))
        seen.add(family);reports.append(dict(case=case['name'],grid=grid,
            nonidentity=desc.get('real_copy_slots',0)))
    result=audit_calls(cases(),calls)
    assert len(seen)==6 and len(calls)==420 and result['original_changed_input_restore_extra_dispatches']==12
    controls=[]
    for name in('grid','capture_key','row_mapping','missing'):
        damaged=copy.deepcopy(calls)
        i=next(i for i,c in enumerate(damaged)if c['copy_grid'])
        if name=='grid':damaged[i]['copy_grid']=1792 if damaged[i]['copy_grid']!=1792 else 128
        elif name=='capture_key':damaged[i]['actual_capture_key']='missing-copy-grid'
        elif name=='row_mapping':damaged[i]['physical_to_original'][0]=-77
        else:damaged.pop()
        try:audit_calls(cases(),damaged)
        except RuntimeError:controls.append(name)
        else:raise AssertionError('corrupt '+name+' not rejected')
    assert 'torch_npu'not in sys.modules
    out=dict(status='HOST_actual_layout_grid_formula_private_key_and_dispatch_audit_passed',
        cases=reports,case_count=408,actual_Native_layout_body_executed=True,
        same1792_indices_and_all_nonidentity_rows_preserved=True,
        original_Native_dispatch_AST_inverse=True,one_key_discriminator_only=True,
        original_capture_validation_replay_bodies_unchanged=True,
        synthetic_graph_execution_records_only=True,expected_cold_families=result['actual_cold_families'],
        expected_positive_math_dispatches=420,expected_extra_restores=12,
        corrupt_record_controls_rejected=controls,NPU_used=False,
        real_model_qualified=False,Goodput=None,adopted=False)
    with(P/'HOST_GRID296_RESULT.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in out.items()if k!='cases'}))
if __name__=='__main__':main()
