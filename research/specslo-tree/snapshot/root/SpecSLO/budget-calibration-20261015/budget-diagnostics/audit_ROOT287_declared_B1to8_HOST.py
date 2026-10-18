"""Original full287 auditor with its two stale B<=5 predicates corrected.

No frozen source/report/receipt is edited. Original child exit1 is retained.
All other startup, sampling, N/C/root limits and numerical gates remain.
"""
import ast,copy,hashlib,inspect,json,sys,types
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-dynamic-root-alternate-eager-compat-full-v287'
O=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
T='dynamic-root-alternate-root-v287-eager3-compatible-full463-ownAR-v1'
A=O/'actual'/T
sys.path.insert(0,str(P))
import check287
sys.path[:0]=[str(P),str(B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'),str(O)]
for name in('tree-high-normal-cap4-native-families-v148','tree-active-window32-16-local-v160',
    'tree-spare-branch-group-budget-full-v260','tree-static-q5-M120-SLO-first-full463-own-AR-v257',
    'tree-static-q5-M120-full463-own-AR-v256','tree-static-q5-M120-node-math-v255',
    'tree-static-q5-M128-full463-own-AR-v251','tree-static-q5-M128-projection-host-v249',
    'tree-static-q5-immutable-index-cache-host-v247','tree-static-q5-long-prefix-origins-node-math-v246',
    'tree-static-q5-projection-host-v243','tree-canonical-contiguous-full463-own-AR-v235',
    'tree-contiguous-branch-tail-final-metadata-math-v234'):
    sys.path.append(str(B/name))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import audit287 as current
    original_functions=current.old.functions
    replacements=[]
    def declared_functions(path,namespace,**kwargs):
        text=original_functions(path,namespace,**kwargs)
        if Path(path).name=='audit_production_v101.py'and kwargs.get('names')==('audit_trace_set',):
            tree=ast.parse(text);before=ast.dump(tree,include_attributes=False);changed=[];functions=[]
            for function in tree.body:
                if not isinstance(function,ast.FunctionDef):continue
                nodes=[n for n in ast.walk(function)if isinstance(n,ast.Compare)and ast.unparse(n)=='1 <= b <= 5']
                if nodes:
                    functions.append(function)
                    for node in nodes:
                        assert isinstance(node.comparators[-1],ast.Constant)and node.comparators[-1].value==5
                        node.comparators[-1].value=8;changed.append(node)
            assert len(changed)==2
            source=ast.unparse(ast.fix_missing_locations(tree))+'\n'
            exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(functions),type_ignores=[])),__file__+'::original-audit-declared-domain8','exec'),namespace)
            for node in changed:node.comparators[-1].value=5
            assert ast.dump(tree,include_attributes=False)==before
            replacements.append(dict(before='1 <= b <= 5',after='1 <= b <= 8',occurrences=2,
                functions=[f.name for f in functions],entire_module_AST_inverse=True,
                original_sha256=hashlib.sha256(text.encode()).hexdigest(),projected_sha256=hashlib.sha256(source.encode()).hexdigest()))
        return text
    proxy=types.ModuleType('_ROOT287_original_audit_namespace_domain8')
    proxy.__dict__.update(vars(current.old));proxy.functions=declared_functions
    truthiness=[]
    def declared_need(value,message):
        if message=='actual online paired gain observations by original generation':
            assert type(value)is dict and set(value)=={'1','2','3','4','5'}
            truthiness.append(dict(message=message,generation_keys=sorted(value),original_value_type='dict',normalization='bool(nonempty actual by_generation dictionary)'))
            return current.need(bool(value),message)
        return current.need(value,message)
    namespace=dict(vars(current),old=proxy,need=declared_need)
    for name in('original_with_new_geometry','audit'):
        fn=getattr(current,name)
        namespace[name]=types.FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__)
    actual=namespace['audit'](A,(48,))
    assert actual['passed']is True and actual['whole_workload_AR_qualified']is True
    assert len(replacements)==1 and len(truthiness)==4
    receipt=json.loads((O/f'receipt-{T}.json').read_text())
    assert receipt['status']=='failed'and receipt['child']['exit_code']==1
    assert receipt['owned_remaining']==[]and receipt['sources_unchanged']is True and receipt['devices_after']['all_idle']is True
    for name in('pre_verifier','post_verifier'):
        row=receipt[name];assert row['exit_code']==0 and sha(row['path'])==row['sha256']
        v=row['actual_result'];assert v['status']=='pass'and v['errors']==[]
        assert(v['checked_frozen_files'],v['checked_external_files'],v['checked_packages'])==(10496,15,212)
    result=dict(status='PASS_external_original287_full_audit_with_declared_B8_domain',
        original_guard_status='failed_child_exit1_stale_auditor_B5_predicate',original_receipt_edited=False,
        original_receipt_sha256=sha(O/f'receipt-{T}.json'),only_raw_auditor_predicate_change=replacements,paired_dictionary_truthiness_correction=truthiness,
        full_original_audit=actual,whole463_AR=True,original_environment_prepost_verified=True,
        owned_cleanup_complete=True,all4_idle_at_terminal=True,formal_final_comparison=False,adopted=False,goal_achieved=False)
    p=D/'ROOT287_INDEPENDENT_FULL_DOMAIN8_AUDIT.json'
    with p.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k!='full_original_audit'}))
if __name__=='__main__':main()
