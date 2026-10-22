import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import torch
torch.set_num_threads(1)
from pathlib import Path
import sys,json,gzip,hashlib,datetime,time
started=time.perf_counter()
w=Path('/root/SpecSLO/budget-calibration-20261015/tree-kernel-verdict-repeatfix-v133');sys.path.insert(0,str(w));d=w/'actual/two-case-teacherPA-repeatfix-v133-1'
from audit_kernel_verdict_v125 import audit_reports
paths=[d/f'rank{i}-kernel-verdict.json.gz' for i in range(4)]
rr=[json.loads(gzip.decompress(p.read_bytes())) for p in paths]
a=audit_reports(rr,private_directory=d);b=json.loads((d/'completed.json').read_text())['actual_component_audit'];diff=[]
def visit(a,b,p):
 if type(a)!=type(b):diff.append(dict(path=p,live_type=type(a).__name__,retained_type=type(b).__name__));return
 if isinstance(a,dict):
  if set(a)!=set(b):diff.append(dict(path=p,kind='dictkeys',difference=list(set(a)^set(b))));return
  for k in a:visit(a[k],b[k],p+'/'+str(k))
 elif isinstance(a,(list,tuple)):
  if len(a)!=len(b):diff.append(dict(path=p,kind='length',live=len(a),retained=len(b)));return
  for i,(x,y) in enumerate(zip(a,b)):visit(x,y,p+'/'+str(i))
 elif a!=b:diff.append(dict(path=p,kind='value',live=repr(a),retained=repr(b)))
visit(a,b,'')
out=dict(schema='ROOT.V133.actual-completion-JSON-boundary-audit.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_strict_auditor_accepts_all_saved_raw=True,differences=diff,canonical_JSON_equal=json.loads(json.dumps(a))==b,elapsed_seconds=time.perf_counter()-started,raw_reports_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},actual_original_audit=a,NPU_query_executed=False,source_files_changed=False,original_supervisor_status='failed',formal_Goodput=None)
p=w.parent/'budget-diagnostics/ROOT_V133_ACTUAL_COMPLETION_JSON_BOUNDARY_AUDIT.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('original_strict_auditor_accepts_all_saved_raw','canonical_JSON_equal','elapsed_seconds')}));print('difference_count',len(diff));print(json.dumps(diff[:12]));print('output',p)
