"""Freeze independent M165/N32 math/cost qualification; never inherit device proof."""
import datetime
import hashlib
import io
import json
import tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
P=B/'tree-M165-N32-independent-target-math-cost-v317'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

def main():
    freeze=P/'FREEZE317.json'
    assert not freeze.exists()
    receipt=B/'tree-causal80-small-cohort-role-aware-audit-full-v209/receipt-measured-target-cost-risk-root-v316-B1to8-full463-ownAR-v1.json'
    completed=read(receipt)
    assert completed['status']=='complete' and not completed['errors'] and completed['sources_unchanged']
    assert not completed['owned_remaining'] and completed['devices_after']['all_idle']
    audit_path=D/'ROOT316_TERMINAL_INDEPENDENT_FULL_AUDIT.json'
    audit=read(audit_path)
    assert audit['status']=='terminal_completed_compute_audited' and not audit['errors']
    assert audit['whole_workload_AR_qualified'] and audit['original_receipt_sha256']==sha(receipt)
    assert read(P/'HOST317_RESULT.json')['cases']==546
    geometry=read(P/'HOST317_GEOMETRY_AND_SPLIT_RESULT.json')
    assert geometry['cases']==546 and geometry['NPU_used']is False
    previous=B/'tree-single-prepare-target-math-cost-v311/FREEZE311.json'
    files=dict(read(previous)['files'])
    files[str(previous)]=sha(previous)
    for directory in (P,B/'tree-M165-single-prepare-host-v313',B/'tree-uniform-q5-M165-S33-C128-host-v285'):
        for path in directory.iterdir():
            if path.is_file() and path.suffix in ('.py','.sh','.json'):
                if path.suffix=='.py':compile(path.read_text(),str(path),'exec')
                if str(path)in files:assert files[str(path)]==sha(path)
                files[str(path)]=sha(path)
    for path in (Path(__file__),receipt,audit_path,D/'ROOT_M165_N32_NEXT_DEVICE_PLAN_20261019.json'):
        files[str(path)]=sha(path)
    for path,wanted in files.items():assert sha(path)==wanted,path
    review=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        geometry='M165/S33/uniformQ5/KV2560/N32/C128/F12',
        preserved_original408_cases=True,declared_neural_cases=546,target_ranks=[1,2,3],
        real_independent_prefills=96,reference_bank_offsets=[32,64],
        synthetic_raw165_controls_per_target=144,all_BF16_raw_words_per_width=65536,
        raw_test_widths=[256,384],slot_dtypes=['int32','int64'],
        paired_target_configurations=16,rotated_rounds=20,
        original_model_precision_and_sampler_and_matchers=True,
        original_startup_inventory_not_run_by_this_component_diagnostic=True,
        fresh_device_qualification_required=True,whole463_AR_qualified=False,
        real_draft_cost_measured=False,Goodput=None,adopted=False)
    with freeze.open('x')as stream:json.dump(dict(schema='ROOT317.frozen-target-math-cost.v1',files=files,review=review,model_qualification_inherited=False),stream,indent=2);stream.write('\n')
    inventory={**files,str(freeze):sha(freeze)}
    archive=P/'SOURCE317.tar.gz'
    with tarfile.open(archive,'x:gz')as saved:
        for path,wanted in sorted(inventory.items()):
            data=Path(path).read_bytes();assert hashlib.sha256(data).hexdigest()==wanted
            member=tarfile.TarInfo(path.lstrip('/'));member.mode=0o600;member.size=len(data);member.mtime=0
            saved.addfile(member,io.BytesIO(data))
    with tarfile.open(archive,'r:gz')as saved:
        assert len(saved.getmembers())==len(inventory)
        for member in saved:assert member.isfile()and hashlib.sha256(saved.extractfile(member).read()).hexdigest()==inventory['/'+member.name]
    pins=dict(freeze_sha256=sha(freeze),archive_sha256=sha(archive),members=len(inventory))
    (P/'SOURCE317_RECEIPT.json').write_text(json.dumps(pins,indent=2)+'\n')
    print(json.dumps(pins))

if __name__=='__main__':main()
