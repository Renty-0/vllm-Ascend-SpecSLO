"""Retained406 records are HOST schema fixtures, never new model qualification."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

P = Path(__file__).resolve().parent
B = P.parent


def rejects(function):
    try:
        function()
    except (RuntimeError, AssertionError):
        return
    raise AssertionError('HOST409 corrupted component proof accepted')


def main():
    sys.path.insert(0, str(B / 'tree-M120-S32-frugal-independent-target-host-v399'))
    from bootstrap399 import install
    install()
    sys.path[:0] = [str(P), str(B / 'tree-S24-S32-physical-dispatch-host-v408')]
    from s32_409 import descriptor, audit, sha
    from scope_wide409 import local_proof, current
    from audit399 import audit as original
    donor = B / ('tree-causal80-small-cohort-role-aware-audit-full-v209/actual/'
                 'frugal-S32-M120-import-fixed-root-v406-200cases-v1')
    expected = original(donor)
    with tempfile.TemporaryDirectory(prefix='HOST409-schema-', dir=P) as temporary:
        outer = Path(temporary)
        root = outer / 'target-S32-independent-math409'
        root.mkdir()
        for rank in range(4):
            value = json.loads((donor / f'rank{rank}-numeric.json').read_text())
            compact = []
            for bank in value['banks']:
                folder = root / f"KV{bank['extent']}"
                folder.mkdir(exist_ok=True)
                filename = f'rank{rank}-bank399.json'
                shutil.copyfile(donor / folder.name / filename, folder / filename)
                compact.append(descriptor(folder, rank, bank))
            value['banks'] = compact
            (root / f'rank{rank}-numeric.json').write_text(json.dumps(value))
        assert audit(root) == expected
        engine = SimpleNamespace(rank=1, is_draft=False, draft_vocab_size=151936,
                                 _v101_output_directory=str(outer))
        shape = (((32, 1, 10, 4096), (32, 32)),)
        entries = {(f'target-{kind}:{vocabulary}|steps:1|tree-fia-full:exact-length-task-update|buffers:{shape}', (120,)):
                   SimpleNamespace(runtime_validated=True)
                   for kind, vocabulary in [('greedy', 151936), ('hidden', 0)]}
        engine.graph_runner = SimpleNamespace(target_entries=entries, disabled_entry_keys=set())
        numeric = root / 'rank1-numeric.json'
        proof = dict(path=str(numeric), sha256=sha(numeric), cases=200,
                     same_model_instance=True, Native_flags_assigned=False)
        engine._S32_409_current_proof = proof
        engine._S32_409_proof_owner = engine
        got, graph_objects = local_proof(engine)
        assert got is proof and graph_objects == entries
        permission = dict(wide_proof=proof, wide_entries=graph_objects)
        assert current(engine, permission)
        first_key = next(iter(entries))
        entries[first_key].runtime_validated = False  # HOST double, no Native entry exists.
        rejects(lambda: local_proof(engine))
        entries[first_key].runtime_validated = True
        value = json.loads(numeric.read_text())
        escaped = copy.deepcopy(value)
        escaped['banks'][0]['path'] = str(donor / 'KV512/rank1-bank399.json')
        numeric.write_text(json.dumps(escaped))
        proof['sha256'] = sha(numeric)
        rejects(lambda: local_proof(engine))
        numeric.write_text(json.dumps(value))
        proof['sha256'] = sha(numeric)
        path = Path(value['banks'][0]['path'])
        bank = json.loads(path.read_text())
        bank['math']['cases'][0]['candidate_KV_copied'] = True
        path.write_text(json.dumps(bank))
        value['banks'][0]['sha256'] = sha(path)
        numeric.write_text(json.dumps(value))
        proof['sha256'] = sha(numeric)
        rejects(lambda: local_proof(engine))
        rejects(lambda: audit(root))
    assert not any(name in sys.modules for name in ('torch', 'torch_npu', 'vllm'))
    result = dict(status='PASS_HOST409_COMPACT_BANK_AUDIT_EQUIVALENCE_AND_SCOPE_CORRUPTIONS',
                  retained406_used_only_as_HOST_JSON_schema_fixture=True,
                  original_finite_audit_exactly_preserved=True,
                  current_path_hash_rank_and_case_checks=True,
                  unvalidated_HOST_graph_double_rejected=True,
                  escaped_old_run_bank_rejected=True,
                  copied_candidate_KV_rejected_even_after_consistent_checksum_update=True,
                  ephemeral_fixture_removed=True,
                  NPU_used=False, model_constructors=0,
                  model_qualification_inherited=False, production_permission=False,
                  Raw=None, Goodput=None)
    (P / 'HOST409_COMPACT_SCOPE_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
