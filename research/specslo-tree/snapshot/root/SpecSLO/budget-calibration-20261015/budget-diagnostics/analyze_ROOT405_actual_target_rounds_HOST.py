"""Bind every retained target call to its actual emitted output chunks."""
from pathlib import Path
import collections
import gzip
import hashlib
import json
import statistics

B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-causal80-small-cohort-role-aware-audit-full-v209/actual/normal24-global64-derived-controls-root-v405-full463-ownAR-v1/windows/window64/full463/rank1-full463-actual-trace.json.gz'
v=json.loads(gzip.decompress(P.read_bytes()))
seen=set();chunks=[];first={};last={}
for e in v['stream_events']:
 i=e['request_index']
 if i in seen:chunks.append(e)
 else:seen.add(i);first[i]=e['elapsed_seconds']
 if e['finished']:last[i]=e['elapsed_seconds']
assert len(first)==len(last)==463
calls=v['normal_target_calls'];assert len(chunks)==sum(c['N']for c in calls)
last_arrival=max(v['absolute_arrivals'])-v['origin']
position=0;prior=0.;rounds=[]
for ordinal,c in enumerate(calls):
 current=chunks[position:position+c['N']];position+=c['N']
 assert [e['request_index']for e in current]==c['sequence_ids'][:c['N']]
 end=max(e['elapsed_seconds']for e in current);assert end>=prior
 active=sum(first[i]<=end<last[i]for i in first)
 rounds.append(dict(ordinal=ordinal,N=c['N'],end_seconds=end,wall_since_previous_chunk_group=end-prior,
  delivered_tokens=sum(len(e['token_ids'])for e in current),B4=sum(b==4 for b in c['budgets']),
  observed_after_first_token_unfinished=active,before_final_arrival=end<last_arrival))
 prior=end

def summarize(rows):
 return dict(calls=len(rows),mean_N=sum(r['N']for r in rows)/len(rows)if rows else None,
  emitted_tokens=sum(r['delivered_tokens']for r in rows),
  wall_between_emitted_groups_seconds=sum(r['wall_since_previous_chunk_group']for r in rows),
  group_gap_median_ms=1000*statistics.median(r['wall_since_previous_chunk_group']for r in rows)if rows else None)
result=dict(status='HOST_ALL2627_TARGET_CALLS_BOUND_TO_ACTUAL_OUTPUT_CHUNKS',source=str(P),
 source_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),calls=len(rounds),output_chunks=len(chunks),
 first_token_events=len(first),last_scheduled_arrival_seconds=last_arrival,
 last_first_token_seconds=max(first.values()),last_completed_seconds=max(last.values()),
 before_final_arrival=summarize([r for r in rounds if r['before_final_arrival']]),
 after_final_arrival=summarize([r for r in rounds if not r['before_final_arrival']]),
 target_N_bands={label:summarize([r for r in rounds if low<=r['N']<=high])for label,low,high in [('1to8',1,8),('9to16',9,16),('17to23',17,23),('24',24,24)]},
 active_after_first_bands={label:summarize([r for r in rounds if low<=r['observed_after_first_token_unfinished']<=high])for label,low,high in [('0to16',0,16),('17to32',17,32),('33to55',33,55),('56to64',56,64)]},
 scope='Wall gaps between emitted target groups include device and host/draft/prefill work; these are not isolated target kernel times. Active count excludes admitted requests that have not emitted a first token.',
 NPU_used=False,new_performance_run=False,optimization_gain_inferred=False)
out=B/'budget-diagnostics/ROOT405_TARGET_ROUND_TIME_DISTRIBUTION_HOST.json'
with out.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2))
