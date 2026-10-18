"""Fresh217 projection: fix original output-leader ownership in own-AR only."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
DONOR = B/'tree-packed-KV-preflight-with-own-FIA-AR-full-v217'
P = B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'
assert not P.exists()
P.mkdir()
records = []
for old in sorted(DONOR.iterdir()):
    if old.suffix not in ('.py', '.sh'):
        continue
    new = P/old.name.replace('217', '218')
    new.write_text(old.read_text().replace(str(DONOR), str(P)).replace('217', '218'))
    records.append(dict(source=str(old), source_sha256=hashlib.sha256(old.read_bytes()).hexdigest(), new=str(new)))


def change(name, before, after):
    p = P/name
    text = p.read_text()
    assert text.count(before) == 1, (name, before)
    p.write_text(text.replace(before, after))


change('reference218.py', "    candidate={r['request_id']:r['completion_token_ids']for r in trace['generate_results']}",
       "    from candidate_comparison218 import load_candidate\n"
       "    candidate,candidate_artifact=stage('AR218/retained-output-leader-comparison-only',\n"
       "        lambda:load_candidate(artifact,trace,engine.topology.target_leader_rank))")
change('reference218.py', '        same_prefill_cohorts=True,reference_trajectories_generated=',
       '        candidate_result_artifact=candidate_artifact,candidate_result_owner_rank=1,\n'
       '        same_prefill_cohorts=True,reference_trajectories_generated=')
change('audit_reference218.py', "    comparisons=[];all_reference=[];metadata_cases=0;canary_cases=0",
       "    leader_path=directory/'windows/window32/full463/rank1-full463-actual-trace.json.gz'\n"
       "    leader_bytes=leader_path.read_bytes();leader=read(leader_path)\n"
       "    leader_info=dict(path=str(leader_path),bytes=len(leader_bytes),sha256=hashlib.sha256(leader_bytes).hexdigest())\n"
       "    need(leader['rank']==1 and type(leader['generate_results'])is list and len(leader['generate_results'])==463,'actual original output-leader-only results')\n"
       "    candidate={r['request_id']:r['completion_token_ids']for r in leader['generate_results']}\n"
       "    need(len(candidate)==463 and set(candidate)=={r['request_id']for r in leader['requests']}and all(len(v)==256 for v in candidate.values()),'complete unique original output-leader results')\n"
       "    comparisons=[];all_reference=[];metadata_cases=0;canary_cases=0")
change('audit_reference218.py', "        trace=read(raw);groups=[r['sequence_ids']for r in trace['standalone_prefill_calls']]",
       "        trace=read(raw)\n"
       "        need(summary['candidate_result_artifact']==leader_info and summary['candidate_result_owner_rank']==1,'exact original output-leader comparison artifact')\n"
       "        for key in('scope_count','manifest_sha256','prefix_sha256','requests','tokenized_inputs','source_generation','standalone_prefill_calls'):\n"
       "            need(trace[key]==leader[key],'independent all-rank input/cohort identity '+key)\n"
       "        need((trace['generate_results']is None)if rank!=1 else(trace['generate_results']==leader['generate_results']),'preserve original only-output-leader result materialization')\n"
       "        groups=[r['sequence_ids']for r in trace['standalone_prefill_calls']]")
change('audit_reference218.py', "            candidate={r['request_id']:r['completion_token_ids']for r in trace['generate_results']}\n", '')
change('guard218.py', 'same216 cost/ownAR plus direct packed targetKV preflight read with fresh actual original/new shadow gate; clients/model/clock unchanged',
       'same217 packed preflight plus rank-aware original output-leader comparison artifact and synchronized own-AR load; clients/model/clock unchanged')
for row in records:
    old = Path(row['source']).read_text().replace(str(DONOR), str(P)).replace('217', '218')
    current = Path(row['new']).read_text()
    row['new_sha256'] = hashlib.sha256(current.encode()).hexdigest()
    row['patch_after_namespace_projection'] = ''.join(difflib.unified_diff(old.splitlines(True), current.splitlines(True)))
    if Path(row['new']).suffix == '.py':
        ast.parse(current)
(P/'PROJECTION218.json').write_text(json.dumps(dict(schema='ROOT218.rank-aware-own-AR-source-projection.v1',
    donor=str(DONOR), donor_freeze_sha256=hashlib.sha256((DONOR/'FREEZE217.json').read_bytes()).hexdigest(),
    projection=records, original_Native_result_ownership_preserved=True,
    candidate_tokens_are_comparison_only_never_reference_inputs=True,
    cold_first_cohort_policy_added=False, NPU_started=False, source_peer_review=False), indent=2)+'\n')
print(json.dumps(dict(status='prepared_unfrozen', directory=str(P), NPU_started=False)))
