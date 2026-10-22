"""Actual438 archive validates the per-token geometry frontier reconstruction."""
from pathlib import Path
import copy,datetime,gzip,hashlib,json,sys,tarfile
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-adaptive-own-history-full-v460';sys.path.insert(0,str(P))
from geometry_history460 import build
info=json.loads((D/'ROOT438_ACTUAL_BATCH_AND_CLASS_OWNER_ROUNDS_HOST.json').read_text());source=Path(info['source_trace']);archival=json.loads((source.parents[3]/'ARCHIVED.json').read_text())
archive=Path(archival['archive']);assert hashlib.sha256(archive.read_bytes()).hexdigest()==archival['archive_sha256']
with tarfile.open(archive,'r:gz')as stream:raw=stream.extractfile(str(source.relative_to(archival['restore_base']))).read()
assert hashlib.sha256(raw).hexdigest()==info['source_SHA256'];trace=json.loads(gzip.decompress(raw))
actual=build(trace,adaptive_required=False)
assert actual['requests']==463 and actual['target_calls']==2484
assert all(row[0]==0 and row[1:]==(120,)*255 for row in actual['schedules'])
controls=[]
for name,edit in[
    ('nonadvancing_frontier',lambda t:t['normal_target_calls'][1]['contexts'].__setitem__(0,t['normal_target_calls'][1]['contexts'][0]-1)),
    ('unsupported_M',lambda t:t['normal_target_calls'][0].__setitem__('physical_M',81)),
    ('publication_count_mismatch',lambda t:t['stream_events'][0]['token_ids'].append(123)),
]:
    changed=copy.deepcopy(trace);edit(changed)
    try:build(changed,adaptive_required=False)
    except RuntimeError:controls.append(name)
    else:raise AssertionError('ROOT460 must reject corrupted actual trace '+name)
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_HOST460_ACTUAL438_PER_OUTPUT_GEOMETRY_RECONSTRUCTION',
    source_trace=str(source),source_trace_sha256=info['source_SHA256'],archive_read_in_place=True,files_restored=0,
    actual_requests=463,actual_target_calls=2484,independent_prediction_positions=463*255,
    actual_frontier_and_publication_counts_agree=True,all_original_geometry120=True,
    corrupted_trace_controls_rejected=controls,production_token_values_used_as_reference_inputs=False,
    NPU_used=False,NN_permission=False,Raw=None,Goodput=None)
with(D/'ROOT460_ACTUAL438_GEOMETRY_HISTORY_HOST_RESULT.json').open('x')as stream:json.dump(out,stream,indent=2);stream.write('\n')
print(json.dumps(out))
