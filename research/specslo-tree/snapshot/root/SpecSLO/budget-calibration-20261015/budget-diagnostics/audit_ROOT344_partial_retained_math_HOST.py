"""Read retained complete NN checkpoints; never repair a failed run's receipt."""
import hashlib
import json
from pathlib import Path
import sys

B = Path(__file__).resolve().parent.parent
P = B/'tree-Q5-uniform-KV-bucket-branch-math-cost-v344'
DONOR = B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG = 'M120-uniform-KV-bucket-root-v344-594-branch-math-current-cost-v1'
sys.path.insert(0, str(P))
import audit344
from domain344 import cases, short_cases, mid_prefixes
from dispatch344 import audit_calls
from runtime344 import original


def main():
    path = DONOR/('receipt-'+TAG+'.json')
    before = path.read_bytes()
    receipt = json.loads(before)
    assert receipt['status'] == 'failed' and 'bounded1800second actual child' in str(receipt['errors'])
    assert receipt['sources_unchanged'] is True and receipt['owned_remaining'] == []
    assert receipt['devices_after']['all_idle'] is True
    assert receipt['post_verifier']['actual_result']['errors'] == []
    fixture = json.loads((original.P/'FIXTURE311.json').read_text())
    heads = []
    rows = []
    root = DONOR/'actual'/TAG
    for label, prefixes, expected in (
            ('mixed', fixture['prefixes'], cases()),
            ('mid-residue-preserved', mid_prefixes(fixture['prefixes']), short_cases())):
        rank_heads = []
        for rank in (1,2,3):
            checkpoint = root/label/f'rank{rank}-math-before-cost.json'
            value = json.loads(checkpoint.read_text())
            assert value['rank'] == rank and value['performance'] is False
            rank_heads.append(audit344.math_rows(value['math'], expected))
            assert value['dispatch'] == audit_calls(expected, value['actual_calls'], prefixes, enabled=True)
            bank = json.loads((root/label/f'rank{rank}-bank-before-math.json').read_text())['evidence']
            assert bank['independent_neural_prefills'] == 72
            assert bank['all_triplet_roots_equal'] is bank['all_triplet_prefix_KV_equal'] is True
            rows.append(dict(bank=label, rank=rank, cases=len(expected),
                independent_prefills=72, complete_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                all_original_and_strict_node_matchers=True, all_prefixes_unchanged_at_end_of_math=True,
                actual_dispatch_reconstructed=True))
        assert rank_heads[0] == rank_heads[1] == rank_heads[2]
        heads.append(len(expected))
    assert path.read_bytes() == before
    assert not list(root.glob('rank*-local-stage-failure*'))
    result = dict(status='PASS_retained_complete_math_checkpoints_ONLY',
        original_receipt_status='failed_deadline_preserved',
        original_receipt_sha256=hashlib.sha256(before).hexdigest(),
        cases_per_target=sum(heads), real_target_roles=3, rows=rows,
        short_bank_case_files_per_target=[len(list((root/'short-residue-preserved/independent-node-math').glob(f'rank{r}-canonical-case*.json')))for r in (1,2,3)],
        short_bank_not_qualified=True, complete_current_cost_report_available=False,
        no_after_cost_prefix_or_normal_exit_qualification=True,
        model_qualification_inherited=False, full594_scope_qualified=False,
        real_draft_generation=False, empirical_g=False, whole463_qualified=False,
        Goodput=None, adopted=False, original_receipt_unchanged=True, NPU_used_by_this_reader=False)
    target = B/'budget-diagnostics/ROOT344_INDEPENDENT_PARTIAL_RETAINED_MATH_AUDIT.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))


if __name__ == '__main__': main()
