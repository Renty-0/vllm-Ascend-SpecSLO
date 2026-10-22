"""Read only the completed same-M proofs; retain failed and correct HOST parses.

This is a partial-qualification explanation, never whole-session completion.
Use Python3.12 once and the retained Python3.11 once with distinct --out paths.
"""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode=True
P=Path('/root/SpecSLO/budget-calibration-20261015/tree-active-cap-controls-v103')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    sys.path.insert(0,str(P))
    from bounded_report_audit_v78 import chain_record
    from lifecycle_actor_v75 import original_admission_sources
    rows=[]
    for rank in range(4):
        path=P/'local-output/active32-64-cap3-4-full463-v1'/f'rank{rank}-same-M-qualification.json.gz'
        raw=path.read_bytes();data=json.loads(gzip.decompress(raw));error=None
        try:chain_record(data,'draft' if rank==0 else 'target',rank=rank)
        except BaseException:error=traceback.format_exc()
        rows.append(dict(rank=rank,path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
            original_chain_record_passed=error is None,error=error,
            raw_lifecycle_source_AST=[r['source_AST'] for r in data['lifecycle']['cases']]))
    value=dict(schema='V103.actual-partial-sameM-HOST-interpreter-execution.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_HOST_argv=[sys.executable,*sys.argv],
        Python=sys.version,source_script_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        expected_source_AST={k:v for k,v in original_admission_sources().items() if k!='functions'},
        actual_four_rank_original_chain_record_passed=all(r['original_chain_record_passed'] for r in rows),
        completed_same_M_artifacts=rows,torch_imported='torch' in sys.modules,NPU_initialized=False,
        whole_session_completion_inferred=False,whole_workload_AR_qualified=False,formal_Goodput=False)
    with Path(args.out).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(output=args.out,Python=sys.version.split()[0],
        actual_four_rank_original_chain_record_passed=value['actual_four_rank_original_chain_record_passed'])))


if __name__=='__main__':main()
