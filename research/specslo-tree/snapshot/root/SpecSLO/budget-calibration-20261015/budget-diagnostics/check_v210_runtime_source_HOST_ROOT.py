"""Probe exact donor projection and spawn callable before any model import."""
import datetime
import hashlib
import importlib.abc
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

D = Path(__file__).resolve().parent
P = D/'v210-source-design-only'


class Firewall(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','acl','transformers'):
            raise ImportError('runtime HOST forbids '+fullname)


def main():
    sys.meta_path.insert(0,Firewall());sys.path.insert(0,str(P))
    root=P/'runtime-HOST-attempts'/str(time.time_ns());root.mkdir(parents=True)
    files=list(P.glob('*.py'))+[Path(__file__)]
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    before={str(p):sha(p) for p in files}
    for p in files:(root/p.name).write_bytes(p.read_bytes())
    result=dict(schema='ROOT.V210.runtime-source-spawn-HOST.v1',status='failed',
                NPU_used=False,Torch_library_imported=False,model_constructed=False,
                production_issuer=False,errors=[])
    try:
        import runtime210
        seen=[]
        def first_gate():
            seen.append('ROOT-source-gate-before-original-ready-or-backend')
            raise RuntimeError('HOST_SOURCE_GATE_BOUNDARY')
        namespace,projection=runtime210.build_runtime(check_source=first_gate)
        assert pickle.loads(pickle.dumps(namespace['worker'])) is runtime210.worker
        class Connection:
            def __init__(self):self.sent=[];self.closed=False
            def send(self,value):self.sent.append(value)
            def close(self):self.closed=True
        conn=Connection()
        namespace['_compiled_worker'](None,0,0,conn,
            diagnostic_spec=None,own_identity=None,output_directory=str(root),admission_windows=(32,))
        assert seen==['ROOT-source-gate-before-original-ready-or-backend']
        assert conn.closed and len(conn.sent)==1 and conn.sent[0][0]=='error'
        assert 'HOST_SOURCE_GATE_BOUNDARY' in conn.sent[0][1]
        saved=json.loads((root/'rank0-last-error.json').read_text())
        assert 'HOST_SOURCE_GATE_BOUNDARY' in saved['error']
        assert before=={p:sha(p) for p in before}
        assert not any(n.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','acl') for n in sys.modules)
        result.update(status='passed_source_boundary_and_spawn_pickle_only',projection=projection,
                      source_before=before,source_after=before,worker_first_source_gate=True,
                      actual_original_exception_and_exit_handler=True,
                      spawn_callable_pickle_roundtrip=True,
                      actual_PEARL_process_spawn_or_model_executed=False)
    except BaseException:result['errors'].append(traceback.format_exc())
    output=root/'RESULT.json'
    with output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(path=str(output),sha256=sha(output),status=result['status'],errors=result['errors'])))
    return 0 if result['status'].startswith('passed_') else 1


if __name__=='__main__':raise SystemExit(main())
