"""Retain the original complete chain before validating its JSON evidence domain.

No tensor/model/collective operation occurs here. The existing all4 stage owns
local file errors. Original save_gzip and all numerical/protocol validators are
unchanged. A later validator failure leaves each completed file intact.
"""
import gzip
import hashlib
import json
from pathlib import Path

SCHEMA='V129.retained-before-validation-structure-evidence.v1'
MAX_COMPRESSED_BYTES=64*1024**2
MAX_UNCOMPRESSED_BYTES=256*1024**2


def require(value,message):
    if not value:raise ValueError('V129 evidence: '+message)


def load_retained(artifact,directory,*,rank):
    require(type(rank) is int and rank in range(4),'exact original rank required')
    require(type(artifact) is dict and set(artifact)=={'path','bytes','uncompressed_bytes','sha256'},'original artifact schema')
    root=Path(directory).resolve();path=Path(artifact['path']).resolve()
    require(path.parent==root and path.name==f'rank{rank}-pre-audit-same-M-qualification.json.gz','original private rank namespace')
    compressed=path.read_bytes()
    require(type(artifact['bytes']) is int and 0<len(compressed)==artifact['bytes']<=MAX_COMPRESSED_BYTES
        and hashlib.sha256(compressed).hexdigest()==artifact['sha256'],'retained compressed bytes/SHA')
    with gzip.open(path,'rb') as stream:raw=stream.read(MAX_UNCOMPRESSED_BYTES+1)
    require(type(artifact['uncompressed_bytes']) is int and len(raw)==artifact['uncompressed_bytes']<=MAX_UNCOMPRESSED_BYTES,'retained JSON expansion/length')
    def reject_constant(value):raise ValueError('V129 evidence: nonfinite JSON '+value)
    value=json.loads(raw,parse_constant=reject_constant)
    require(type(value) is dict,'complete chain JSON object required')
    return value,hashlib.sha256(raw).hexdigest()


def retain_structure_evidence(engine,chain,*,stage_all4):
    """Return a chain read from the actual saved gzip, never a synthetic copy."""
    from production_probe_v101 import save_gzip
    rank=engine.rank
    def retain():
        require(type(rank) is int and rank in range(4),'exact original rank required')
        directory=Path(engine._v101_output_directory).resolve()
        require(directory.is_dir(),'original actual output directory required')
        return save_gzip(directory,f'rank{rank}-pre-audit-same-M-qualification.json.gz',chain)
    artifact=stage_all4('V129-retain-complete-structure-before-validation',retain)
    value,raw_sha=stage_all4('V129-reload-retained-structure-JSON-domain',lambda:
        load_retained(artifact,engine._v101_output_directory,rank=rank))
    evidence=dict(schema=SCHEMA,rank=rank,artifact=artifact,uncompressed_sha256=raw_sha,
        actual_saved_gzip_reloaded=True,retained_before_complete_validator=True,
        in_memory_tuple_or_integer_key_domain_used_for_validation=False,
        numerical_or_protocol_validator_modified=False,NPU_qualification_inferred_from_retention=False)
    return value,evidence


def audit_structure_evidence(evidence,chain,directory,*,rank):
    require(type(evidence) is dict and evidence.get('schema')==SCHEMA and evidence.get('rank')==rank
        and type(evidence.get('rank')) is int,'retention evidence rank/schema')
    require(evidence.get('actual_saved_gzip_reloaded') is evidence.get('retained_before_complete_validator') is True
        and all(evidence.get(k) is False for k in ('in_memory_tuple_or_integer_key_domain_used_for_validation',
            'numerical_or_protocol_validator_modified','NPU_qualification_inferred_from_retention')),'retention must not grant numerical qualification')
    value,raw_sha=load_retained(evidence['artifact'],directory,rank=rank)
    require(raw_sha==evidence['uncompressed_sha256'] and value==chain,'full retained/pre-audit and reported chain differ')
    return dict(passed=True,rank=rank,artifact=evidence['artifact'],uncompressed_sha256=raw_sha,
        retention_precedes_original_complete_validator=True,numerical_qualification_inferred=False)


def source_contract():
    """Only retention/readback/metadata differ from the source-bound parent."""
    root=Path(__file__).resolve().parent
    parent=json.loads((root/'V129_PARENT_SOURCE.json').read_text())
    old=Path(parent['directory'])
    unchanged=('bounded_report_audit_v78.py','boundary_schema_v101.py','native_boundaries.py',
        'original_hook_adapter.py','production_probe_v101.py','local_qualification_scope.py',
        'local_mixed_qualification.py','independent_history_v75.py','normal_draft_actor_v75.py',
        'combined_controls_v113.py','budget_policy.py','startup_config_v75.py')
    for name in unchanged:
        require((root/name).read_bytes()==(old/name).read_bytes(),'original numerical/protocol/production helper changed: '+name)
    edits={
        'shape_scope_v101.py':("    chain['actual_inventory']=dict(before=before,after=after)\n",
            "    from serialized_structure_evidence_v129 import retain_structure_evidence\n    chain,evidence=retain_structure_evidence(engine,chain,stage_all4=stage_all4)\n    engine._v129_structure_evidence=evidence\n"),
        'combined_runtime_v113.py':("                    report['S35_structural_scope']=scope_report(structural_scope,engine)\n",
            "                    report['structure_evidence_retention']=engine._v129_structure_evidence\n"),
        'audit_combined_v113.py':("        chain=original.load_artifact(proof['artifact'],private_directory);finite_json(chain)\n",
            "        from serialized_structure_evidence_v129 import audit_structure_evidence\n        audit_structure_evidence(report['structure_evidence_retention'],chain,private_directory,rank=rank)\n")}
    for name,(anchor,addition) in edits.items():
        original=(old/name).read_text()
        require(original.count(anchor)==1 and (root/name).read_text()==original.replace(anchor,anchor+addition),
            'execution-source delta exceeded exact retention/readback/metadata insertions: '+name)
    for name in ('launch_combined_v113.sh','combined-startup-body.sh'):
        original=(old/name).read_text()
        require(original.count(str(old))==1 and (root/name).read_text()==original.replace(str(old),str(root)),
            'launcher delta exceeded exact task-root rebind')
    return dict(original_complete_validator_bytes_unchanged=True,original_Native_protocol_and_numerics_unchanged=True,
        complete_chain_saved_before_validation=True,real_saved_JSON_readback_required=True,
        original_21scope26phase_required=True,retention_is_not_qualification=True)
