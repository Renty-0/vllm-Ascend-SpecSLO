"""Original supervision flow with explicit process/device/verifier HOST stubs."""
import datetime
import hashlib
import importlib.abc
import io
import contextlib
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace as NS

D=Path(__file__).resolve().parent;P=D/'v210-source-design-only'


class Firewall(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend','acl','transformers'):
            raise ImportError('guard HOST forbids '+name)


def main():
    sys.meta_path.insert(0,Firewall());sys.path.insert(0,str(P))
    root=P/'guard-HOST-attempts'/str(time.time_ns());root.mkdir(parents=True)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files=list(P.glob('*.py'))+list(P.glob('*.sh'))+[Path(__file__)]
    before={str(p):sha(p)for p in files}
    for p in files:(root/p.name).write_bytes(p.read_bytes())
    result=dict(schema='ROOT.V210.original-supervision-control-HOST.v1',status='failed',
                NPU_used=False,model_constructed=False,Torch_library_imported=False,
                actual_process_or_device_query_executed=False,errors=[])
    try:
        import guard210
        guard210.EXPECTED_FREEZE='a'*64
        guard210.EXPECTED_ARCHIVE='b'*64
        cases=[]
        for label,child_code,bad_pre,bad_source in [('success',0,False,False),
                ('child-fails',1,False,False),('pre-verifier-fails',0,True,False),
                ('source-post-mismatch',0,False,True)]:
            folder=root/label;folder.mkdir();ns=guard210.execution_namespace();observed=[]
            source=folder/'SOURCE-carrier.txt';source.write_text('explicit HOST input')
            expected='0'*64 if bad_source else sha(source)
            ns.update(P=folder,prepared=lambda:({}, {str(source):expected},dict(HOST_stub=True)),
                      storage=lambda tag=None:dict(actual_tag_bytes=0,aggregate_private_bytes=0))
            def save(path,value):Path(path).write_text(json.dumps(value))
            def verify(tag,phase):
                observed.append('verify-'+phase)
                path=folder/(tag+'-'+phase+'-verifier.log')
                row=dict(status='fail'if bad_pre and phase=='pre'else'pass',
                         errors=['explicit HOST rejection']if bad_pre and phase=='pre'else[],
                         checked_frozen_files=10496,checked_external_files=15,checked_packages=212)
                path.write_text(json.dumps(row)+'\n')
                return dict(path=str(path),exit_code=1 if bad_pre and phase=='pre' else 0)
            def idle():observed.append('device-HOST-stub');return dict(all_idle=True,HOST_stub=True)
            def cleanup(child,helper,row):observed.append('cleanup-HOST-stub');row['owned_remaining']=[]
            ns['original_helpers']=lambda:dict(save=save,verify=verify,
                helper=lambda:NS(device_state=idle),owned_cleanup=cleanup)
            class Process:
                pid=123;returncode=child_code
                def poll(self):return self.returncode
                def wait(self):return self.returncode
            def popen(*args,**kwargs):observed.append('Popen-HOST-stub');return Process()
            ns['subprocess']=NS(Popen=popen,STDOUT=-2)
            ns['actual_audit']=lambda directory:({'passed':True,'HOST_stub':True},'HOST_only',{})
            with contextlib.redirect_stdout(io.StringIO()):
                try:ns['execute_locked']('HOST-'+label)
                except SystemExit as error:assert error.code==1
            receipt=json.loads((folder/('receipt-HOST-'+label+'.json')).read_text())
            assert receipt['status']==('complete'if label=='success'else'failed')
            assert 'cleanup-HOST-stub'in observed and 'verify-post'in observed
            assert receipt['owned_remaining']==[]and receipt['devices_after']['all_idle']is True
            assert ('Popen-HOST-stub'in observed)is(not bad_pre)
            assert receipt['sources_unchanged']is(not bad_source)
            assert receipt['formal_Goodput']is receipt['full463']is False
            assert ('actual_execution_complete'in receipt)is(label in('success','source-post-mismatch'))
            cases.append(dict(case=label,status=receipt['status'],observed=observed,
                              receipt=str(folder/('receipt-HOST-'+label+'.json'))))
        assert before=={p:sha(p)for p in before}
        assert not any(n.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend','acl')for n in sys.modules)
        result.update(status='passed_original_control_flow_with_explicit_HOST_stubs',cases=cases,
                      source_before=before,source_after=before,
                      interpretation='Control/exception/metadata coverage only; no baseline, device, owned process, source-freeze or numerical qualification.')
    except BaseException:result['errors'].append(traceback.format_exc())
    output=root/'RESULT.json'
    with output.open('x')as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(path=str(output),sha256=sha(output),status=result['status'],errors=result['errors'])))
    return 0 if result['status'].startswith('passed_') else 1


if __name__=='__main__':raise SystemExit(main())
