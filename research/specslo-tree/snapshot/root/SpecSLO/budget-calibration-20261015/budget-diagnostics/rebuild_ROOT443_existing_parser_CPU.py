"""Recreate the byte-identical private CPU parser from retained source hashes."""
from pathlib import Path
import hashlib,json,shutil
D=Path('/root/SpecSLO/budget-calibration-20261015/budget-diagnostics')
P=D/'ROOT443_PRIVATE_CANN_PARSER'
r=json.loads((D/'ROOT443_PRIVATE_PARSER_MANIFEST.json').read_text())
assert not P.exists()
for v in r['files']:
    src=Path(v['original']);dst=Path(v['copy'])
    assert dst.is_relative_to(P) and hashlib.sha256(src.read_bytes()).hexdigest()==v['sha256']
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,dst)
    dst.chmod(0o755 if dst.name in('msprof','msprof.py')else 0o644)
for p in P.rglob('*'):
    if p.is_dir():p.chmod(0o755)
print(json.dumps(dict(status='REBUILT_PRIVATE_BYTE_IDENTICAL_EXISTING_CPU_PARSER',
    original_runtime_modified=False,NPU_used=False,installed_new_environment=False)))
