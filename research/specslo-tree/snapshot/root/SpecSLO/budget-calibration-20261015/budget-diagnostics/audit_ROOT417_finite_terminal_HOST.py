"""Independently reread the finite399/guard417 terminal without model imports."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

B = Path('/root/SpecSLO/budget-calibration-20261015')
DONOR = B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
P = B/'tree-M120-S30-independent-crossS24-finite-v417'
FREEZE = '35efd13830927ed9cfa24cbe6e9e0fc5310b8a77e7a4c057b81dea51e299720a'
ARCHIVE = '7f7b42a19d2431365c20ab0bf2d0342555ac582685fdf793ccc787733e85629f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(value, message):
    if not value:
        raise AssertionError('ROOT417 terminal: '+message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--expected-receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    need('/' not in args.tag and not args.output.exists(), 'fresh bounded output/tag')
    receipt = DONOR/f'receipt-{args.tag}.json'
    need(sha(receipt) == args.expected_receipt_sha256, 'exact actual terminal receipt')
    value = json.loads(receipt.read_text())
    need(value['schema'] == 'ROOT.V399.finite-frugal-S32-independent-S24-supervisor.v1'
         and value['supervision_metadata_repair400'] is True
         and value['startup_import_repair406'] is True
         and value['new_component_version'] == 417 and value['new_numerical_geometry'] is True
         and value['status'] in ('complete', 'failed'), 'actual new417 with original owned supervision')
    need(value['sources_unchanged'] and not value['owned_remaining'] and value['devices_after']['all_idle'],
         'all frozen sources and owned cleanup/all4 idle')
    for phase in ('pre_verifier', 'post_verifier'):
        evidence = value[phase]
        need(evidence['exit_code'] == 0 and evidence['actual_result']['status'] == 'pass'
             and sha(evidence['path']) == evidence['sha256'], 'actual original '+phase)
    sys.path.insert(0, str(P))
    import guard417
    guard417.EXPECTED_FREEZE, guard417.EXPECTED_ARCHIVE = FREEZE, ARCHIVE
    _, sources, _ = guard417.prepared()
    actual = DONOR/'actual'/args.tag
    result = dict(status='RETAINED_FAILED_FINITE_RUN', receipt=str(receipt),
                  receipt_sha256=args.expected_receipt_sha256, source_members=len(sources),
                  source_freeze_sha256=FREEZE, source_archive_sha256=ARCHIVE,
                  supervisor_status=value['status'], supervisor_errors=value['errors'],
                  complete_finite_math=False, cost_panels=None, NPU_used=False,
                  full463=False, whole_AR=False, production_permission=False,
                  Raw=None, Goodput=None, adopted=False, goal_achieved=False)
    if value['status'] == 'complete':
        sys.path.insert(0, str(B/'tree-M120-S30-independent-crossS24-finite-v417'))
        from audit417 import audit
        audited = audit(actual)
        need(audited == value['actual_finite399_audit'], 'two retained finite audits agree')
        panels = []
        for extent, rows in audited['costs'].items():
            need(len(rows) == 3, 'all three actual target cost ranks')
            old = statistics.median(row['median_ms']['S24'] for row in rows)
            new = statistics.median(row['median_ms']['S30'] for row in rows)
            panels.append(dict(KV_extent=int(extent), S24_graph_median_ms=old,
                               S30_graph_median_ms=new, S30_over_S24_graph_cost=new/old,
                               S24_max_rank_p95_ms=max(row['p95_ms']['S24'] for row in rows),
                               S30_max_rank_p95_ms=max(row['p95_ms']['S30'] for row in rows)))
        result.update(status='COMPLETE_FINITE200_STRICT_M120_S30_CROSS_M120_S24_MATH_AND_GRAPH_COST',
                      complete_finite_math=True, cases_each_target=200,
                      audited_finite=audited, cost_panels=sorted(panels,key=lambda row:row['KV_extent']),
                      cost_scope='prepared physical graph calls; excludes proposal building, serving traffic and draft/target overlap',
                      global_active_limit_required=64, candidate_M=120, independent_reference_M=120,
                      candidate_S=32, independent_reference_S=24)
    else:
        result['retained_role_errors'] = {path.name:json.loads(path.read_text())
                                          for path in actual.glob('rank*-last-error.json')}
    need(not any(name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for name in sys.modules),
         'HOST terminal reader imports no model or NPU backend')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('status','supervisor_status','complete_finite_math','cost_panels')}))


if __name__ == '__main__':
    main()
