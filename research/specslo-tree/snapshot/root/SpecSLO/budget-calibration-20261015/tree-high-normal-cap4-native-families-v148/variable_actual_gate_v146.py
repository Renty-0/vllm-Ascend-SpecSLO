"""Prior local component evidence is necessary; it never grants full527 scope."""
import json
from pathlib import Path
from source_snapshot_v132 import sha


def validate_documents(receipt,audit,proof):
    if (receipt.get('status')!='complete' or receipt.get('errors')!=[] or receipt.get('owned_remaining')!=[]
        or receipt.get('sources_unchanged') is not True or receipt.get('actual_execution_complete') is not True
        or receipt['devices_before']['all_idle'] is not True or receipt['devices_after']['all_idle'] is not True
        or receipt['pre_verifier']['exit_code']!=0 or receipt['post_verifier']['exit_code']!=0
        or receipt['child']['exit_code']!=0 or audit.get('status')!='passed'
        or audit['receipt']['sha256']!=proof['receipt_sha256']
        or audit['component_numerical_qualified'] is not True or audit['full527_or_full463_qualified'] is not False
        or audit['exploration_commits']!=8 or audit['fixed_protocol_replay_commits']!=16
        or audit['next_draft_probes']!=6 or audit['complete_local_protocol_zero_traces']!=2
        or audit['natural_protocol_repair_coverage']!={'0':True,'1':True}
        or audit['actual_config_scope']!=dict(maxseq=32,maxqueued=64,storage=96,block=128,pages=544,dual_bank_lease_pages=140)):
        raise RuntimeError('V146 requires the complete independently audited V144 local component; it does not grant full527 scope')


def require_actual_variable_component(configuration):
    proof=configuration.get('component_actual_evidence',{}).get('variable_B4_local')
    if not isinstance(proof,dict) or set(proof)!={'receipt','receipt_sha256','external_audit','external_audit_sha256'}:
        raise RuntimeError('V146 variable-B4 actual component evidence is pending; HOST source readiness cannot launch a model')
    receipt_path,audit_path=Path(proof['receipt']),Path(proof['external_audit'])
    if sha(receipt_path)!=proof['receipt_sha256'] or sha(audit_path)!=proof['external_audit_sha256']:
        raise RuntimeError('V146 prior actual component receipt/audit identity differs')
    receipt=json.loads(receipt_path.read_text());audit=json.loads(audit_path.read_text())
    validate_documents(receipt,audit,proof)
    parent=receipt_path.parent
    frozen=json.loads((parent/'PREPARED_SOURCE_HASHES_V144.json').read_text())
    identity=audit['source_identity']
    if (sha(parent/'PREPARED_SOURCE_HASHES_V144.json')!=identity['freeze_sha256']
        or sha(parent/'CPU_RESULT_V144.json')!=identity['CPU_sha256']
        or sha(parent/'PREPARED_SOURCE_ARCHIVE_V144.tar.gz')!=identity['prepared_archive']['sha256']):
        raise RuntimeError('V146 prior local component CPU/freeze/archive identity differs')
    for name,wanted in {**frozen['files'],**frozen['external_files']}.items():
        if sha(name)!=wanted:raise RuntimeError('V146 prior local component bound source changed: '+name)
    strict=audit['original_strict_audit']
    for name,wanted in {**strict['raw_checkpoint_sha256'],**strict['raw_report_sha256']}.items():
        if sha(name)!=wanted:raise RuntimeError('V146 prior local actual raw evidence changed: '+name)
    return dict(proof=proof,prior_local_component_qualified=True,prior_scope=dict(maxseq=32,maxqueued=64,storage=96),
        grants_current_full527_qualification=False,current_full_process_component_required=True,
        current_full_two65zero_and_full463_still_required=True)
