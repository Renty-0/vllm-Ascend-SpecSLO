"""Re-execute private-tail proofs with actual Native allocator and PA padding."""
import ast,hashlib,importlib.util,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
def main():
    sys.path[:0]=[str(P),str(B/'tree-private-alternate-draft-tail-host-v271')]
    for name in('check271_HOST','check_metadata271_HOST'):
        path=B/'tree-private-alternate-draft-tail-host-v271'/(name+'.py')
        spec=importlib.util.spec_from_file_location('_ROOT277_'+name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.P=P;module.B=B;module.main()
    geometry=json.loads((P/'HOST271_RESULT.json').read_text())
    metadata=json.loads((P/'HOST_METADATA271_RESULT.json').read_text())
    assert geometry['cases']==680 and metadata['cases']==75
    for path in P.glob('*.py'):compile(path.read_text(),str(path),'exec')
    source=(P/'copy277.py').read_text();t=ast.parse(source)
    call=next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='copy_rows')
    imports=[n for n in ast.walk(call)if isinstance(n,ast.ImportFrom)and n.module=='copy263']
    assert len(imports)==1 and [a.name for a in imports[0].names]==['_copy_kv_rows263']
    assert 'width!=1024'in source and 'ROW_BYTES=2048,PAIRS=1792,COL_TILE=2048'in source
    assert 'torch_npu'not in sys.modules
    result=dict(status='HOST_private_draft_geometry_and_original_metadata_passed',
        symbolic_cases=680,original_PA_metadata_cases=75,
        raw_copy_kernel_body_reused=True,new_specialization_width=1024,
        declared_N=[1,2,4,8,12],declared_steps=[1,2,3],declared_variants=[0,1,0],
        actual_NN_cases_required=45,actual_prefix_count=24,
        original_freeze_inputs_edited=False,NPU_used=False,
        neural_qualified=False,whole_workload_AR_qualified=False,Goodput=None,adopted=False)
    with(P/'HOST277_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
