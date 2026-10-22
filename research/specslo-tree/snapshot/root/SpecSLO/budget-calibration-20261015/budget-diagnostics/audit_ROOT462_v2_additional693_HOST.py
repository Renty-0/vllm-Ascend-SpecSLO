"""Read every current693 NN record while subsequent untimed checks continue."""
from pathlib import Path
import datetime
import hashlib
import importlib.abc
import json
import sys

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-adaptive-qualified-wait-margin-full-v462'
TAG = 'adaptive-own-history-global64-root-v462-qualified-wait-full463-ownAR-v2'
ROOT = B / 'tree-causal80-small-cohort-role-aware-audit-full-v209/actual' / TAG


class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend'):
            raise RuntimeError('HOST693 reader cannot import model/device backends')


sys.meta_path.insert(0, NoModel())
sys.path.insert(0, str(P))
import check462
from oracle459 import cases
from dispatch377 import audit_calls


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    frozen = check462.ready('9f79697914d74f79cc94fe4605c8f20665c44ee2f6b33fe99b59b73d7b4e2280')
    expected = cases()
    assert len(expected) == 231
    heads = []
    evidence = {}
    summaries = []
    for rank in range(4):
        path = ROOT / 'adaptive-own-physical-node-math460' / f'rank{rank}-proof460.json'
        proof = json.loads(path.read_text())
        evidence[str(path)] = sha(path)
        assert proof['cases'] == 693 and proof['same_model_instance'] is True
        assert proof['global_active_ceiling'] == 64 and proof['Native_flags_assigned'] is False
        assert proof['parameter_SHA_before'] == proof['parameter_SHA_after']
        assert proof['parameter_SHA_before'] and proof['original_branch_copy_state_installed_once'] is True
        assert proof['old75_plus6_still_required'] is True and proof['production_permission'] is False
        assert [bank['name'] for bank in proof['banks']] == ['mixed', 'mid', 'short']
        rank_heads = []
        for bank in proof['banks']:
            path = Path(bank['path']).resolve()
            assert path.is_relative_to(ROOT) and sha(path) == bank['sha256']
            evidence[str(path)] = sha(path)
            data = json.loads(path.read_text())
            rows = data['math']['cases']
            assert data['rank'] == rank and len(rows) == 231 and bank['cases'] == 231
            if rank == 0:
                assert data['bank'] is None and not data['actual_calls']
                assert all(row is None for row in rows)
                continue
            assert data['bank']['independent_neural_prefills'] == 72
            assert data['bank']['all_triplet_roots_equal'] is True
            assert data['bank']['all_triplet_prefix_KV_equal'] is True
            assert data['math']['all_independent_prefixes_unchanged'] is True
            prefixes = [[0] * count for count in data['prefix_lengths']]
            actual_dispatch = audit_calls(expected, data['actual_calls'], prefixes)
            # The real auditor returns tuple families; saved JSON represents
            # those same tuples as lists. Compare the complete wire value.
            assert json.loads(json.dumps(actual_dispatch)) == data['dispatch']
            for case, row in zip(expected, rows, strict=True):
                assert row['case'] == case and row['exact_greedy'] is True
                assert row['hidden_bitwise_equal'] is True and row['KV_bitwise_equal'] is True
                assert row['all128_KV_original_matcher_passed'] is True
                assert row['graph_eager_KV_bitwise_equal'] is True
                assert row['candidate_KV_copied'] is False
                assert row['candidate_tokens'] == row['reference_tokens']
                assert all(frame['physical_queries'] == case['M']
                           for frame in row['independent_reference_frames'])
                rank_heads.append(row['candidate_tokens'])
            summaries.append(dict(rank=rank, bank=bank['name'], cases=231,
                                  actual_dispatch=actual_dispatch, independent_prefills=72,
                                  all72_prefixes_unchanged=True))
        if rank:
            assert len(rank_heads) == 693
            heads.append(rank_heads)
    assert heads[0] == heads[1] == heads[2]
    result = dict(
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='PASS_HOST462_V2_ALL693_ACTUAL_SAME_PROCESS_OWN_PHYSICAL_NN_RECORDS',
        tag=TAG, original_source_freeze='9f79697914d74f79cc94fe4605c8f20665c44ee2f6b33fe99b59b73d7b4e2280',
        own_frozen_files=len(frozen['files']), cases_each_target=693,
        independent_prefills_each_target=216, all3_predictions_agree=True,
        all_hidden_and_all128_KV_bitwise_equal=True,
        all_graph_eager_KV_equal=True, all_parameter_SHA_unchanged=True,
        Native_flags_assigned=False, summaries=summaries, actual_evidence_SHA256=evidence,
        old75_plus6_still_required=True, full463=False,
        whole_workload_AR_qualified=False, Raw=None, Goodput=None,
        current_NPU_process_unchanged=True, HOST_NPU_used=False,
        HOST_model_constructors=0, production_permission=False)
    with (D / 'ROOT462_V2_ADDITIONAL693_INDEPENDENT_NN_READER_HOST.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('summaries', 'actual_evidence_SHA256')}), flush=True)


if __name__ == '__main__':
    main()
