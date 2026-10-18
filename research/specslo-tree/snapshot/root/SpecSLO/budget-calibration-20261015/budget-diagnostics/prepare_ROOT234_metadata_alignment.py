"""Fresh copy of233 with final Native mask metadata saved before replay."""
import ast,json
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');OLD=B/'tree-contiguous-branch-tail-M80-math-v233';CORE=B/'tree-canonical-M80-FULL9-full463-own-AR-v231';P=B/'tree-contiguous-branch-tail-final-metadata-math-v234'
P.mkdir(exist_ok=False)
for p in OLD.iterdir():
    if p.suffix in('.py','.sh'):(P/p.name.replace('233','234')).write_text(p.read_text().replace(str(OLD),str(P)).replace('233','234'))
for name in('branch_tail231.py','check_metadata_state231_CPU.py'):(P/name).write_bytes((CORE/name).read_bytes())
for name in('HOST_MAPPER231_RESULT.json','CPU_REFERENCE_SCHEDULE231_RESULT.json','CPU_BANK_PADDING231_RESULT.json'):(P/name).write_bytes((OLD/name).read_bytes())
(P/'FIXTURE234.json').write_bytes((OLD/'FIXTURE233.json').read_bytes())
p=P/'baseline234.py';s=p.read_text();a='    def original_layout(plans,ids,inputs,pos,meta):';assert s.count(a)==1;s=s.replace(a,a+'\n        from fixed_mask_extent import padded\n        meta=padded(meta,9)');p.write_text(s)
p=P/'ROOT_SOURCE_REVIEW234.json';p.write_text(json.dumps(dict(status='source_prepared_device_unrun',same155_strict_neural_cases=True,
    repair='no-copy State metadata and original-layout negative-control metadata explicitly use the same final mask capacity9 as Native forward; copy mode remains10',
    actual_Native_forward_CPU_metadata_identity_test_required=True,old233_failure_preserved=True,original_Native_tolerance_and_ABI_gate_unchanged=True,
    original_sampler_unchanged=True,weights_changed=False,production_qualified=False,performance_qualified=False,adopted=False),indent=2)+'\n')
for p in P.glob('*.py'):ast.parse(p.read_text())
print(json.dumps(dict(directory=str(P),prepared=True,NPU_started=False)))
