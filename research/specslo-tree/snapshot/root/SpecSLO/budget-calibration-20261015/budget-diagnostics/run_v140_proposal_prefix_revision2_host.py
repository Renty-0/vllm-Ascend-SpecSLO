"""Small retained-3.11 HOST check of the external proposal-prefix correction."""
import ast
from collections import Counter
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

D=Path(__file__).resolve().parent
P=D.parent/'tree-normal-admission-fix-v140'
OLD=D/'analyze_v140_actual_normal_admission.py'
NEW=D/'analyze_v140_actual_normal_admission_revision2.py'
FAILURE=D/'ROOT_V140_ACTUAL_AUDIT_ATTEMPT1_EXTERNAL_FAILURE.json'
OUTPUT=D/'V140_EXTERNAL_PROPOSAL_PREFIX_REVISION2_HOST.json'


def desc(path):
    data=path.read_bytes()
    return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    started=time.perf_counter()
    assert sys.version_info[:2]==(3,11)
    assert os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']=='0'
    assert os.environ['OMP_NUM_THREADS']==os.environ['OPENBLAS_NUM_THREADS']=='1'
    before={str(p):desc(p) for p in (OLD,NEW,FAILURE)}
    assert before[str(OLD)]['sha256']=='6243e164c361ab0ab0dd307fc8a7f1e6804025fb7bf80aabb6597bc12e69615e'
    old=load(OLD,'old_external_v140');new=load(NEW,'new_external_v140')
    a,b=ast.parse(OLD.read_text()),ast.parse(NEW.read_text())
    for tree in (a,b):tree.body=[n for n in tree.body if not(isinstance(n,ast.FunctionDef) and n.name=='proposal_counts')]
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    out=P/'local-output/normal-admission-v140-one-full463-v1'
    report_path=out/'rank1-local-diagnostic.json'
    envelope=json.loads(report_path.read_bytes())['traces'][0]
    artifact=envelope['artifact'];path=Path(artifact['path']);compressed=path.read_bytes()
    assert len(compressed)==artifact['bytes'] and hashlib.sha256(compressed).hexdigest()==artifact['sha256']
    raw=gzip.decompress(compressed);assert len(raw)==artifact['uncompressed_bytes']
    trace=json.loads(raw);assert trace['role']=='warmup65'
    call=trace['normal_target_calls'][0]
    assert call==dict(N=1,budgets=[5],contexts=[86],family='full9',physical_M=40,
        physical_S=8,physical_query_partition=[6,5,5,5,5,5,5,4],real_query_count=6,
        sequence_ids=[0,463,464,465,466,467,468,469])
    rows=[json.loads(line) for line in new.MANIFEST.read_bytes().splitlines()[:65]]
    tests=[];rejections=[]

    def reject(module,record,label):
        try:module.proposal_counts(record,rows)
        except ValueError as error:rejections.append(dict(label=label,message=str(error)));return
        raise AssertionError('accepted corrupt or legacy-bug record: '+label)

    base=dict(normal_target_calls=[call],logical_events=[])
    reject(old,base,'old immutable auditor rejects genuine physical padding')
    assert new.proposal_counts(base,rows)==Counter({(rows[0]['slo_class'],5):1})
    unpadded=next(c for c in trace['normal_target_calls'] if c['N']==c['physical_S'])
    expected=Counter((rows[i]['slo_class'],b) for i,b in zip(unpadded['sequence_ids'],unpadded['budgets'],strict=True))
    assert new.proposal_counts(dict(normal_target_calls=[unpadded],logical_events=[]),rows)==expected
    tests.append('genuine first-warm padded and genuine unpadded real-prefix counts')

    for label,key,value in [('wrong N','N',2),('wrong budget length','budgets',[5,1]),
                            ('out-of-domain B','budgets',[0])]:
        changed=copy.deepcopy(base);changed['normal_target_calls'][0][key]=value;reject(new,changed,label)
    tests.append('N and real budget corruption rejected')

    for label,index,value in [('wrong scratch prefix',1,470),('real owner in scratch suffix',1,1),
                             ('real owner is scratch',0,463)]:
        changed=copy.deepcopy(base);changed['normal_target_calls'][0]['sequence_ids'][index]=value;reject(new,changed,label)
    tests.append('dedicated scratch suffix and real-owner domain corruption rejected')

    for label,key,value in [('wrong partition','physical_query_partition',[5,6,5,5,5,5,5,4]),
                            ('wrong M','physical_M',41),('wrong S','physical_S',7),
                            ('wrong real Q','real_query_count',7),('wrong contexts','contexts',[])]:
        changed=copy.deepcopy(base);changed['normal_target_calls'][0][key]=value;reject(new,changed,label)
    tests.append('physical partition M/S and real Q/context corruption rejected')

    mixed=dict(normal_target_calls=[],logical_events=[dict(kind='plan',target_ids=[0,1],budgets=[3,1])])
    assert new.proposal_counts(mixed,rows)==Counter([(rows[0]['slo_class'],3),(rows[1]['slo_class'],1)])
    changed=copy.deepcopy(mixed);changed['logical_events'][0]['target_ids'].append(463);reject(new,changed,'mixed owner suffix cannot be sliced away')
    tests.append('mixed real-owner exact-length rule retained')
    assert len(tests)==5
    assert before=={str(p):desc(p) for p in (OLD,NEW,FAILURE)}
    forbidden=[name for name in sys.modules if name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','numpy')]
    assert forbidden==[]
    result=dict(schema='V140.external-proposal-prefix-revision2.HOST.v1',status='passed',
        actual_argv=sys.orig_argv,tests=tests,test_count=len(tests),rejections=rejections,
        actual_fixture=dict(report=desc(report_path),artifact=artifact,role=trace['role'],call_index=0,record=call),
        source_delta=dict(only_function_changed='proposal_counts',all_other_AST_identical=True),
        inputs_before=before,inputs_after=before,runner=desc(Path(__file__).resolve()),
        elapsed_seconds=time.perf_counter()-started,old_failed_attempt_preserved=True,
        auditor_main_executed=False,full_actual_audit_reexecuted=False,
        frozen_sources_or_receipt_changed=False,Torch_imported=False,NPU_initialized=False,
        model_or_query_executed=False,physical_cards_requeried=False,forbidden_modules=forbidden)
    with OUTPUT.open('x') as stream:stream.write(json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps(dict(status='passed',test_count=len(tests),output=desc(OUTPUT)),sort_keys=True))


if __name__=='__main__':main()
