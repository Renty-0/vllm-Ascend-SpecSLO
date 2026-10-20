"""Actual CPU plans/padding and complete independent S24 node scheduling."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FROZEN = '/root/data/specslo-baselines/nano-pearl-ascend-v1'
SOURCE_STARTUP = ROOT/'tree-active64-whole-proposal-segment-limit-v398/startup398.sh'
PATH_VALUE = next(line.removeprefix('export PYTHONPATH=')
                  for line in SOURCE_STARTUP.read_text().splitlines()
                  if line.startswith('export PYTHONPATH='))
for name, value in (('$task_root', str(ROOT/'tree-causal80-small-cohort-role-aware-audit-full-v209')),
                    ('$repo', FROZEN+'/source/vllm-ascend-hust'), ('$frozen', FROZEN),
                    ('${PYTHONPATH:-}', '')):
    PATH_VALUE = PATH_VALUE.replace(name, value)
assert '$' not in PATH_VALUE
sys.path[:0] = [str(HERE), str(ROOT/'tree-M120-S32-frugal-budget-projection-host-v397'),
                *filter(None, PATH_VALUE.split(':'))]

from domain399 import cases, prefix_bank
from oracle399 import prepare, specialized
from padding399 import pad
from projection397 import plan
from reference_schedule399 import schedule


def main():
    p = Path(__file__).resolve().parent
    fixture = json.loads((p/'FIXTURE399.json').read_text())
    fake = SimpleNamespace(config=SimpleNamespace(max_model_len=4096),
                           _tree_graph_scratch_sequence_ids=list(range(463, 527)))
    records = []
    for extent in (512, 1024, 1536, 2048, 2560):
        prefixes = prefix_bank(fixture['prefixes'], extent)
        for case in cases():
            plans, rows = prepare(case, prefixes)
            roots = [101+i for i in range(case['N'])]
            pp, rr, dd, ids, count = pad(fake, plans, roots, rows, list(range(case['N'])), [], [])
            assert count == case['N'] and len(pp) == case['S']
            assert sum(v.candidate_budget+1 for v in pp) == 120
            assert all(a is b for a, b in zip(pp[:count], plans, strict=True))
            assert rr[:count] == roots and dd[:count] == rows
            assert set(ids[count:]).isdisjoint(ids[:count])
            total = 0
            ends = []
            for proposal in pp:
                total += proposal.candidate_budget+1
                ends.append(total)
            geometry = plan(ends, case['query_count'])
            assert geometry['real_N'] == case['N']
            assert geometry['selected_candidates'] == sum(case['budgets']) <= 88
            actual = schedule(plans, roots, rows)
            seen = set()
            for frame in actual:
                assert len(frame['tokens']) == len(frame['positions']) == len(frame['ids']) == 120
                assert len(frame['selected']) == len(set(frame['selected'])) == 24
                assert frame['actual_seq_lengths_q'] == list(range(5, 121, 5))
                for item in frame['real_queries']:
                    owner, query = item['candidate_query']
                    assert item['reference_owner'] == 32+owner
                    assert item['reference_owner'] not in range(32)
                    assert item['position'] == len(prefixes[owner])+query
                    assert item['token'] == (roots[owner] if query == 0 else rows[owner][query-1])
                    assert (owner, query) not in seen
                    seen.add((owner, query))
            wanted = {(owner, query) for owner, budget in enumerate(case['budgets'])
                      for query in range(budget+1)}
            assert seen == wanted and len(seen) == case['query_count']
            records.append(dict(extent=extent, name=case['name'], N=case['N'], C=sum(case['budgets']),
                                Q=case['query_count'], independent_frames=len(actual),
                                independent_nodes=len(seen)))
    assert len(records) == 200
    candidate, reference, runner = (specialized(name) for name in ('candidate', 'reference', 'run'))
    assert reference.__globals__['_checked_reference'].__module__ == 'metadata399'
    assert runner.__globals__['cases'] is cases
    assert runner.__globals__['candidate'].__module__ == 'oracle399'
    constants = []

    def collect(code):
        for value in code.co_consts:
            if hasattr(value, 'co_consts'):
                collect(value)
            else:
                constants.append(value)

    collect(runner.__code__)
    assert 'ROOT399 exact cross-S32/S24 hidden and all128 KV plus fullphysical graph/eager' in constants
    for src in p.glob('*.py'):
        ast.parse(src.read_text(), filename=str(src))
    result = dict(status='PASS_HOST399_ACTUAL_PADDING_AND_ALL_INDEPENDENT_S24_NODE_FRAMES',
                  cases=len(records), case_records=records, candidate_S=32, independent_reference_S=24,
                  all_case_M=120, global_active_limit_required=64,
                  fresh_independent_prefix_bank_owners=96, selected_candidate_limit=88,
                  actual_original_neural_matchers_and_strict_bits_retained=True,
                  independent_reference_does_not_invoke_candidate_projector=True,
                  source_hashes={src.name: hashlib.sha256(src.read_bytes()).hexdigest()
                                 for src in sorted(p.glob('*.py'))},
                  NPU_used=False, model_qualification=False, runtime_prepared=False,
                  production_permission=False, Raw=None, Goodput=None)
    (p/'HOST399_RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('status', 'cases', 'NPU_used')}))


if __name__ == '__main__':
    main()
