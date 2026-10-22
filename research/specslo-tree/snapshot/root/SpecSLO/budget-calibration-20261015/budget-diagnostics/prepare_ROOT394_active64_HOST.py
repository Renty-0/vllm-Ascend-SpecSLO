"""Build additive active64 successor; original376 bytes remain immutable."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

B=Path('/root/SpecSLO/budget-calibration-20261015')
OLD=B/'tree-best257-uniform-KV-buckets-full-v376'
P=B/'tree-best257-uniform-KV-buckets-full-v394'
BASE=B/'tree-static-q5-M120-SLO-first-full463-own-AR-v257'
PARENT_SHA='ae723ddfddeb6d47f04605a3842de0c3690196b0addf39fa17c5bd1ca248765c'

def replace_once(text,before,after):
    assert text.count(before)==1,before
    return text.replace(before,after)

def cap64(text):
    text=re.sub(r'\b48\b','64',text)
    return text.replace('active48','active64').replace('window48','window64')

def main():
    assert hashlib.sha256((OLD/'FREEZE376.json').read_bytes()).hexdigest()==PARENT_SHA
    P.mkdir(exist_ok=False)
    derivation=[]
    for path in OLD.iterdir():
        if path.suffix not in ('.py','.sh'):continue
        text=path.read_text().replace('376','394')
        target=P/path.name.replace('376','394')
        if path.name in ('runtime376.py','guard376.py'):text=cap64(text)
        if path.name=='trace376.py':text=text.replace('window257','window394')
        if path.name=='runtime376.py':text=text.replace('from window257 import parse_windows','from window394 import parse_windows')
        if path.name=='audit376.py':text=replace_once(text,'import audit257 as original','import audit_base394 as original')
        if path.name=='extension376.py':
            text=replace_once(text,'postfull_AR = original.postfull_AR',"def postfull_AR(engine, native, stage, windows, output):\n    from reference394 import run\n    return run(engine, native, stage, windows, output)")
            text=text.replace('original_best257_admission_allocator_draft_and_branch_copy_preserved=True','original_best257_algorithm_preserved_except_global_active_limit=True, global_active_limit=64, previous_global_active_limit=48')
        if path.name=='check376.py':
            text=replace_once(text,"expected = expected or os.environ.get('SPECSLO_V394_ROOT_FREEZE_SHA')", "expected = expected or read(P / 'SOURCE394_RECEIPT.json')['freeze_sha256']")
            text=replace_once(text,"for filename in ('HOST394_RESULT.json', 'HOST394_SCOPE_RESULT.json', 'HOST394_CONSTRUCTION_RESULT.json'):","for filename in ('HOST394_RESULT.json', 'HOST394_SCOPE_RESULT.json', 'HOST394_CONSTRUCTION_RESULT.json', 'HOST394_CAPACITY_RESULT.json'):")
        if path.name=='guard376.py':
            text=replace_once(text,"'SPECSLO_V394_ROOT_FREEZE_SHA='+EXPECTED_FREEZE", "'SPECSLO_V376_ROOT_FREEZE_SHA="+PARENT_SHA+"'")
            text=text.replace('original_best257_optimized_draft_unchanged=True, full463=True,','original_best257_optimized_draft_unchanged=True, active_request_limit=64, previous_active_request_limit=48, full463=True,')
        if path.name=='launch376.sh':
            text=replace_once(text,'  SPECSLO_V394_ROOT_FREEZE_SHA="${SPECSLO_V394_ROOT_FREEZE_SHA:?explicit ROOT394 freeze required}" \\\n','  SPECSLO_V376_ROOT_FREEZE_SHA='+PARENT_SHA+' \\\n')
        target.write_text(text)
        derivation.append(dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),output=target.name,output_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    for before,after in [('window257.py','window394.py'),('empty257.py','empty394.py'),('reference257.py','reference394.py'),('audit_reference257.py','audit_reference394.py')]:
        source=BASE/before;text=cap64(source.read_text());(P/after).write_text(text)
        derivation.append(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),output=after,output_sha256=hashlib.sha256((P/after).read_bytes()).hexdigest()))
    text=(BASE/'audit257.py').read_text().replace('window257','window394').replace('empty257','empty394').replace('audit_reference257','audit_reference394')
    (P/'audit_base394.py').write_text(text)
    for path in P.iterdir():
        if path.suffix=='.py':ast.parse(path.read_text(),filename=str(path))
        if path.suffix=='.sh':subprocess.run(['bash','-n',str(path)],check=True)
    (P/'DERIVATION394.json').write_text(json.dumps(dict(parent376_freeze_sha256=PARENT_SHA,declared_delta='active request ceiling48to64; matching cohort/reference/audit windows; no target math, draft or budget-policy change',files=derivation,NPU_used=False),indent=2)+'\n')
    print(json.dumps(dict(path=str(P),created_files=len(list(P.iterdir())),NPU_used=False)))

if __name__=='__main__':main()
