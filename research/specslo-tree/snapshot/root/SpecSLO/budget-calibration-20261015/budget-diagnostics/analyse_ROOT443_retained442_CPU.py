"""Parse copied actual device capture outside daemon workers, no NPU use."""
from pathlib import Path
import datetime,hashlib,json,sys,time,os

D=Path('/root/SpecSLO/budget-calibration-20261015/budget-diagnostics')
manifest=json.loads((D/'ROOT443_OFFLINE442_INPUT_MANIFEST.json').read_text())
def check_inputs():
    for v in manifest['files']:
        for key in ('original','copy'):
            p=Path(v[key])
            assert p.stat().st_size==v['bytes']
            assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
    assert hashlib.sha256(Path(manifest['receipt']).read_bytes()).hexdigest()==manifest['receipt_sha256']
check_inputs()
private_parser=D/'ROOT443_PRIVATE_CANN_PARSER/tools/profiler/bin'
assert private_parser.is_dir()
os.environ['PATH']=str(private_parser)+':'+os.environ['PATH']
import torch,torch_npu
def blocked(*a,**k):raise RuntimeError('ROOT443 offline parser must not initialize an NPU')
torch.npu._lazy_init=blocked;torch_npu.npu._lazy_init=blocked
from torch_npu.profiler.profiler import analyse
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage='analyse_copied_actual442',NPU_used=False)),flush=True)
profiles=sorted((D/'ROOT443_OFFLINE442_PROFILE_INPUT').glob('rank*/*_ascend_pt'))
assert len(profiles)==3
for path in profiles:
    analyse(str(path),max_process_number=1)
check_inputs()
root=D/'ROOT443_OFFLINE442_PROFILE_INPUT'
csvs=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    for p in sorted(root.rglob('*.csv'))]
if not csvs:raise RuntimeError('ROOT443 no analysed actual device CSV produced')
size=sum(p.stat().st_size for p in root.rglob('*')if p.is_file())
if size>48*1024**2:raise RuntimeError('ROOT443 unchanged private48MiB offline-output cap')
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='ACTUAL442_DEVICE_CAPTURE_PARSED_OFFLINE_CPU',
    original_receipt_sha256=manifest['receipt_sha256'],original442_status_unchanged='failed',
    original_inputs_unchanged=True,NPU_used=False,NN_permission_issued=False,
    bytes=size,csvs=csvs,Raw=None,Goodput=None,Git_pushes=0)
with(D/'ROOT443_OFFLINE442_PARSE_RESULT.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out),flush=True)
