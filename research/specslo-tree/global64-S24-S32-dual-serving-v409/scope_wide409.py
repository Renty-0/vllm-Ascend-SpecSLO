"""Additional current-engine S32 proof; never edits Native qualification flags."""
import json
from pathlib import Path

from s32_409 import sha, load_banks


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT409 wide scope: ' + message)


def local_proof(engine):
    proof = getattr(engine, '_S32_409_current_proof', None)
    need(proof is not None and getattr(engine, '_S32_409_proof_owner', None) is engine
         and proof['cases'] == 200 and proof['same_model_instance'] is True
         and proof['Native_flags_assigned'] is False, 'fresh current engine200 proof')
    root = Path(engine._v101_output_directory).resolve() / 'target-S32-independent-math409'
    path = Path(proof['path']).resolve()
    need(path == root / f'rank{engine.rank}-numeric.json' and sha(path) == proof['sha256'],
         'current role numeric descriptor and exact digest')
    value = load_banks(json.loads(path.read_text()), root)
    need(value['rank'] == engine.rank and len(value['banks']) == 5
         and value['actual_model_cases'] == (0 if engine.is_draft else 200), 'all five actual role banks')
    from domain399 import cases
    from geometry409 import query_ends_wide
    expected = cases()
    need([bank['extent'] for bank in value['banks']] == [512, 1024, 1536, 2048, 2560],
         'all exact predeclared visible extents')
    for bank in value['banks']:
        need(len(bank['math']['cases']) == 40, 'all forty shapes per extent')
        if engine.is_draft:
            need(all(row is None for row in bank['math']['cases']), 'draft has no target numerical claim')
            continue
        prefix = bank['independent_bank']
        need(prefix['independent_neural_prefills'] == 96
             and prefix['all_triplet_roots_equal'] is True
             and prefix['all_triplet_prefix_KV_equal'] is True
             and bank['all96_prefixes_protected'] is True
             and bank['math']['all_independent_prefixes_unchanged'] is True,
             'all independent candidate/reference prefixes protected')
        for case, row in zip(expected, bank['math']['cases'], strict=True):
            need(row['case'] == case and row['candidate_KV_copied'] is False
                 and row['actual_physical_S'] == 32 and row['independent_reference_S'] == 24
                 and row['actual_physical_M'] == 120
                 and row['candidate_tokens'] == row['reference_tokens']
                 and all(row[key] is True for key in (
                     'exact_greedy', 'hidden_bitwise_equal', 'KV_bitwise_equal',
                     'graph_eager_KV_bitwise_equal', 'all128_KV_original_matcher_passed',
                     'allphysical_graph_eager_hidden_KV')), 'all strict independent S32 node and KV gates')
        need(len(bank['actual_static_graph_calls']) >= 40, 'actual graph dispatches retained')
        for call in bank['actual_static_graph_calls']:
            need(call['M'] == 120 and call['attention_segments'] == 32
                 and call['actual_seq_lengths_q'] == query_ends_wide()
                 and call['sequence_lens'] == [bank['extent']] * 32
                 and call['used_aclgraph'] is True and call['copy_enabled'] is False,
                 'actual complete M120/S32 physical dispatch')
    entries = {}
    if not engine.is_draft:
        shape = (((32, 1, 10, 4096), (32, 32)),)
        for kind, vocabulary in [('greedy', engine.draft_vocab_size), ('hidden', 0)]:
            key = (f'target-{kind}:{vocabulary}|steps:1|tree-fia-full:exact-length-task-update|buffers:{shape}', (120,))
            entry = engine.graph_runner.target_entries.get(key)
            need(entry is not None and entry.runtime_validated is True
                 and key not in engine.graph_runner.disabled_entry_keys,
                 'actual naturally validated S32 greedy and hidden graph objects')
            entries[key] = entry
    return proof, entries


def current(engine, permission):
    need(permission['wide_proof'] is getattr(engine, '_S32_409_current_proof', None)
         and getattr(engine, '_S32_409_proof_owner', None) is engine,
         'same current completed S32 proof owner')
    if not engine.is_draft:
        for key, entry in permission['wide_entries'].items():
            need(engine.graph_runner.target_entries.get(key) is entry
                 and entry.runtime_validated is True
                 and key not in engine.graph_runner.disabled_entry_keys,
                 'same actual validated S32 graph entries')
    return True
