"""Recheck actual archived438 own-AR through the dimension-generalized auditor."""
from pathlib import Path
import datetime,gzip,hashlib,importlib.abc,json,sys,tarfile
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-adaptive-own-history-full-v460';R=B/'tree-causal80-small-cohort-role-aware-audit-full-v209';sys.path.insert(0,str(P))
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):
            raise RuntimeError('HOST460 original AR audit no backend')
sys.meta_path.insert(0,NoModel())
from host_env460 import setup
setup()
source=D/'ROOT438_ACTUAL_BATCH_AND_CLASS_OWNER_ROUNDS_HOST.json';info=json.loads(source.read_text());trace=Path(info['source_trace']);actual=trace.parents[3]
archival=json.loads((actual/'ARCHIVED.json').read_text());archive=Path(archival['archive'])
assert hashlib.sha256(archive.read_bytes()).hexdigest()==archival['archive_sha256']
index=json.loads(gzip.decompress(Path(archival['full_index']).read_bytes()))['per_file']
memory={}
with tarfile.open(archive,'r|gz')as stream:
    for member in stream:
        if not member.isfile():continue
        name=member.name
        wanted='/postfull-independent-FIA-AR/'in name or('/windows/window64/full463/'in name and Path(name).name in[f'rank{r}-full463-actual-trace.json.gz'for r in range(4)])
        if not wanted:continue
        raw=stream.extractfile(member).read()
        assert len(raw)==index[name]['bytes']and hashlib.sha256(raw).hexdigest()==index[name]['sha256']
        memory[str(R/name)]=raw
class ArchivePath(type(Path())):
    def read_bytes(self):
        if str(self)not in memory:raise RuntimeError('HOST460 unindexed archived read '+str(self))
        return memory[str(self)]
def read(path):
    path=ArchivePath(path);raw=path.read_bytes()
    return json.loads(gzip.decompress(raw)if path.suffix=='.gz'else raw)
import audit_reference460
body=audit_reference460.constructed(adaptive_required=False)
body.__globals__.update(Path=ArchivePath,read=read)
result=body(str(actual))
assert result['references_per_target']==463 and result['output_tokens_per_target']==118528
assert result['same_backend_exact_output_qualified']is True and len(result['by_target_rank'])==3
victim=next(name for name in memory if name.endswith('/rank1-cohort0.json.gz'))
old=memory[victim];changed=read(victim);changed['steps'][0]['metadata']['exact_mask_sha256']='0'*64
memory[victim]=gzip.compress(json.dumps(changed).encode(),mtime=0)
try:
    try:body(str(actual))
    except RuntimeError as error:assert'attention visibility mask'in str(error)
    else:raise AssertionError('HOST460 corrupted actual attention mask must fail')
finally:memory[victim]=old
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_HOST460_ACTUAL438_FULL_AR_OBLIGATIONS_RETAINED_IN_COMPATIBILITY_CONTROL',
    original438_supervisor_status='failed',original438_failure_not_rewritten=True,actual_old_data_only=True,
    original_archive=str(archive),original_archive_sha256=archival['archive_sha256'],
    archived_members_SHA_verified=len(memory),files_restored=0,archive_read_in_place=True,
    references_each_target=463,output_tokens_each_target=118528,all3_actual_original_AR_outputs_equal=True,
    metadata_cases=result['metadata_cases'],causal_future_canary_cases=result['causal_future_canary_cases'],
    original_all21_Native_counters_zero_verified=True,corrupted_actual_attention_mask_rejected=True,
    compatibility_mode_requires_no_adaptive_policy_claim=True,
    current460_NN_or_AR_qualified=False,NPU_used=False,NN_permission=False,Raw=None,Goodput=None)
with(D/'ROOT460_OLD438_AR_AUDITOR_COMPATIBILITY_HOST_RESULT.json').open('x')as stream:json.dump(out,stream,indent=2);stream.write('\n')
print(json.dumps(out))
