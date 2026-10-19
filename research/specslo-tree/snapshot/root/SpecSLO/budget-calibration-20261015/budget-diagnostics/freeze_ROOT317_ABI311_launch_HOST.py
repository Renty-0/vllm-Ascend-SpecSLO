"""Bind an additive startup ABI filename repair without changing frozen317 math."""
import hashlib
import io
import json
import tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-M165-N32-independent-target-math-cost-v317'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    original=P/'FREEZE317.json'
    assert sha(original)=='36cbac2bca38b3b69b2ab4ed04f984bb48782fd51d5bb682d04f3d98ba76405f'
    files=dict(json.loads(original.read_text())['files']);files[str(original)]=sha(original)
    failed=B/'tree-causal80-small-cohort-role-aware-audit-full-v209/receipt-M165-N32-root-v317-B1to8-546-math-cost-v1.json'
    row=json.loads(failed.read_text());assert row['status']=='failed'and not row['owned_remaining']and row['devices_after']['all_idle']and row['sources_unchanged']
    for name in ('guard317_abi311.py','launch317_abi311.sh','startup317_abi311.sh','ABI311_LAUNCH_REPAIR_REVIEW.json'):
        path=P/name;files[str(path)]=sha(path)
    files[str(Path(__file__))]=sha(__file__);files[str(failed)]=sha(failed)
    for path,wanted in files.items():assert sha(path)==wanted,path
    freeze=P/'FREEZE317_ABI311.json'
    with freeze.open('x')as f:json.dump(dict(schema='ROOT317.additive-ABI311-launch-repair.v1',files=files,numerical_body_unchanged=True,qualification_inherited=False),f,indent=2);f.write('\n')
    inventory={**files,str(freeze):sha(freeze)}
    archive=P/'SOURCE317_ABI311.tar.gz'
    with tarfile.open(archive,'x:gz')as saved:
        for path,wanted in sorted(inventory.items()):
            data=Path(path).read_bytes();assert hashlib.sha256(data).hexdigest()==wanted
            item=tarfile.TarInfo(path.lstrip('/'));item.mode=0o600;item.size=len(data);item.mtime=0
            saved.addfile(item,io.BytesIO(data))
    with tarfile.open(archive,'r:gz')as saved:
        assert len(saved.getmembers())==len(inventory)
        for item in saved:assert item.isfile()and hashlib.sha256(saved.extractfile(item).read()).hexdigest()==inventory['/'+item.name]
    pins=dict(freeze_sha256=sha(freeze),archive_sha256=sha(archive),members=len(inventory))
    (P/'SOURCE317_ABI311_RECEIPT.json').write_text(json.dumps(pins,indent=2)+'\n');print(json.dumps(pins))
if __name__=='__main__':main()
