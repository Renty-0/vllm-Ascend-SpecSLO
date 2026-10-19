"""Prepare common Q11 mask capacity, keeping all real visibility unchanged."""
from pathlib import Path
import ast
import json

B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-M165-common-mask11-target-math-cost-v323'
OLD=B/'tree-M165-N32-independent-target-math-cost-v317'
BASE=B/'tree-uniform-q5-M165-S33-C128-host-v285'

def changed(source,replacements):
    text=source.read_text()
    for before,after in replacements:
        if text.count(before)!=1:
            raise RuntimeError('exact common-mask anchor '+str(source)+' '+before)
        text=text.replace(before,after)
    return text

def main():
    P.mkdir(exist_ok=False)
    (P/'static_base323.py').write_text(changed(BASE/'static_base285.py',[
        ('mask_capacity=11 if state.enabled else 10','mask_capacity=11'),
        ('linear_mask_capacity=10','linear_mask_capacity=11')]))
    text=(B/'tree-M165-single-prepare-host-v313/static_state313.py').read_text().replace('static_base285','static_base323')
    (P/'static_state323.py').write_text(text)
    (P/'frames323.py').write_text(changed(BASE/'frames285.py',[('meta = padded(meta, 10)','meta = padded(meta, 11)')]))
    text=(BASE/'metadata285.py').read_text().replace('from frames285 import pack','from frames323 import pack').replace('from static_base285 import ensure_zero','from static_base323 import ensure_zero').replace('(33, 1, 10, 4096)','(33, 1, 11, 4096)')
    (P/'metadata323.py').write_text(text)
    for name in ('oracle','baseline','benchmark','probes'):
        text=(OLD/f'{name}317.py').read_text()
        for before,after in [('static_state317','static_state323'),('oracle317','oracle323'),('baseline317','baseline323'),('audit_static_calls317','audit_static_calls323'),('benchmark317','benchmark323')]:
            text=text.replace(before,after)
        if name=='oracle':
            assert text.count("'metadata231':'metadata285'")==1
            text=text.replace("'metadata231':'metadata285'","'metadata231':'metadata323'")
        (P/f'{name}323.py').write_text(text)
    text=changed(OLD/'audit_static_calls317.py',[
        ("with_name('FIXTURE317.json')","with_name('FIXTURE323.json')"),
        ('capacity=11 if branch else 10','capacity=11'),
        ('{(10,0),(11,128)','{(11,0),(11,128)')])
    (P/'audit_static_calls323.py').write_text(text)
    (P/'FIXTURE323.json').write_bytes((OLD/'FIXTURE317.json').read_bytes())
    text=(BASE/'fixture285_HOST.py').read_text().replace('from static_base285 import install','from static_base323 import install')
    text=text[:text.index('    from layout285 import layout\n')]+"    import branch_tail285\n    from static_state313 import bind_checked_prepare\n    bind_checked_prepare(engine._branch_tail_state,branch_tail285)\n    return env\n"
    (P/'fixture323_HOST.py').write_text(text)
    text=(OLD/'check_geometry317_HOST.py').read_text()
    text=text.replace("'fixture313_HOST'","'fixture323_HOST'")
    text=text.replace("from oracle317 import cases","from oracle323 import cases")
    text=text.replace("FIXTURE317.json","FIXTURE323.json").replace('HOST317_GEOMETRY_AND_SPLIT_RESULT.json','HOST323_GEOMETRY_AND_SPLIT_RESULT.json')
    anchor="    outer=next(n for n in function.body if isinstance(n,ast.For))"
    assert text.count(anchor)==1
    text=text.replace(anchor,"    for node in ast.walk(function):\n        if isinstance(node,ast.IfExp)and ast.unparse(node)=='11 if any((b > 4 for b in budgets)) else 10':node.orelse=ast.Constant(11)\n"+anchor)
    (P/'check_geometry323_HOST.py').write_text(text)
    for source in P.glob('*.py'):
        ast.parse(source.read_text())
    review=dict(status='SOURCE_ONLY_common_mask_capacity11',reason='Fresh322 trace: first root divergence at layer0 attention for independent side root; all128 prefix/alias tensors raw16 equal. NZ166 does not resolve it.',
        delta='same Q11 allocated mask capacity for candidate primary/branch and independent reference; real queries5,33groups,165rows,visible keys and all KV positions unchanged',
        all546_cases_and_original_matchers_retained=True,new_NPU_qualification_required=True,performance=None,Goodput=None)
    (P/'SOURCE323_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n')
    print(json.dumps(review))

if __name__=='__main__':main()
