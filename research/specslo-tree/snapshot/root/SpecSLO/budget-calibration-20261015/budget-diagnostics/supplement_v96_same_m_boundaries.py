"""Supplement only V96's retained same-M proof; preserve its first audit."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

import analyze_v96_actual_capacity64 as original
from actual_boundary_schema_host import audit_raw_boundaries,isolated_chain_validator

ORIGINAL_AUDIT_SHA='cc7d7e15c73276a95856ab4b00f28bdd512404df918d86dc8908d44c4cb9b2e1'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    original.require(sys.dont_write_bytecode,'Use retained Python3.11 -B')
    original.require(__import__('re').fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.tag),'Invalid tag')
    base=original.BASE/'budget-diagnostics';source=original.SOURCE;directory=source/'local-output'/args.tag
    prior_path=base/f'ROOT_V96_ACTUAL_CAPACITY64_{args.tag}_AUDIT.json'
    original.require(original.file_sha(prior_path)==ORIGINAL_AUDIT_SHA,'Original one-time actual audit changed')
    prior=original.read_json(prior_path)
    original.require(len(prior['errors'])==1 and prior['errors'][0]['section']=='same_M_numerical_and_transaction_qualification',
                     'Supplement limited to the known complete-boundary schema rejection')
    out=base/f'ROOT_V96_ACTUAL_SAME_M_BOUNDARY_SUPPLEMENT_{args.tag}.json'
    original.require(not out.exists(),'Refusing to replace an actual supplemental audit')
    report=dict(schema='ROOT.V96.actual-same-M-complete-boundary-supplement.v1',tag=args.tag,
        UTC=datetime.now(timezone.utc).isoformat(),original_actual_audit=original.descriptor(prior_path),
        original_actual_audit_error_preserved=prior['errors'],original_actual_status=prior['original_status'],
        execution=dict(argv=list(sys.argv),executable=sys.executable,python_version=sys.version,
                       model_or_NPU_imports_allowed=False,device_work=False),errors=[],ranks=[],
        same_M_local_five_phase_qualification=False,full_qualification=False,
        whole_workload_AR_equality_qualified=False,online_qualified=False,formal_Goodput=False)
    prior_artifacts={a['path']:a for a in prior['artifacts']}
    freeze=original.read_json(source/'PREPARED_SOURCE_HASHES_V96.json')
    original.require(original.file_sha(source/'PREPARED_SOURCE_HASHES_V96.json')==original.FREEZE_SHA,
                     'Frozen V96 source identity changed')
    signatures=[]
    with original.frozen_host_modules(source) as modules:
        validator=modules['bounded_report_audit_v78']
        original.require(original.file_sha(validator.__file__)==freeze['files'][validator.__file__],
                         'Original numerical validator bytes changed')
        check,provenance=isolated_chain_validator(validator);report['narrow_schema_adaptation']=provenance
        for rank in range(4):
            try:
                rp=directory/f'rank{rank}-local-diagnostic.json'
                original.require(original.file_sha(rp)==prior_artifacts[str(rp)]['sha256'],'Actual worker report changed')
                worker=original.read_json(rp);pointer=worker['mixed_same_M_qualification']['artifact']
                original.require(pointer['sha256']==prior_artifacts[pointer['path']]['sha256'],'Actual same-M descriptor changed')
                chain=original.read_gzip(pointer,directory)
                before=hashlib.sha256(json.dumps(chain,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                signature=check(chain,'draft' if rank==0 else 'target',rank=rank);signatures.append(signature)
                rows=[audit_raw_boundaries(row,scope,'draft' if rank==0 else 'target',rank)
                      for row,scope in zip(chain['scopes'],modules['local_qualification_scope'].scopes(),strict=True)]
                after=hashlib.sha256(json.dumps(chain,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                original.require(before==after,'Supplement modified actual raw numerical/event record')
                report['ranks'].append(dict(rank=rank,passed=True,artifact=pointer,scopes=rows,
                    original_all_numerical_MC2_Native_compaction_commit_next_draft_lifecycle_checks_passed=True,
                    raw_chain_unchanged=True,semantic_signature_sha256=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()))
            except Exception as error:
                report['errors'].append(dict(rank=rank,error=type(error).__name__+': '+str(error),traceback=traceback.format_exc()))
    if signatures and any(row!=signatures[0] for row in signatures[1:]):
        report['errors'].append(dict(error='Actual four-rank semantic commit/publication signatures differ'))
    if len(report['ranks'])==4:
        for index in range(2):
            common=[r['scopes'][index]['common_signature_sha256'] for r in report['ranks']]
            if len(set(common))!=1:report['errors'].append(dict(error='Complete raw common-boundary sequences disagree across ranks'))
    report['same_M_local_five_phase_qualification']=len(report['ranks'])==4 and not report['errors']
    report['scope']='Only the actual two source-bound N15/N16 FULL197/S19 scopes/five phases; original matcher/exact heads/all28+64 query KV/Native changed-restored/128route MC2/compaction/commit/publication/next-draft/four lifecycle controls. Raw complete boundary logs audited by kind, never changed. Whole-workload AR and performance goals remain unqualified.'
    report['self']=original.descriptor(Path(__file__));report['boundary_helper']=original.descriptor(Path(__file__).with_name('actual_boundary_schema_host.py'))
    with out.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    out.chmod(0o600)
    print(json.dumps(dict(output=str(out),sha256=original.file_sha(out),errors=len(report['errors']),
        same_M_local_five_phase_qualification=report['same_M_local_five_phase_qualification'],formal_Goodput=False)))
    return int(not report['same_M_local_five_phase_qualification'])


if __name__=='__main__':raise SystemExit(main())
