"""Four-role bounded diagnostic, same model math; no online performance run."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import sys
import traceback

P=Path(__file__).resolve().parent;B=P.parent
OLD=B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'
DONOR=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
sys.path[:0]=[str(P),str(OLD),str(DONOR)]
for f in ['tree-private-alternate-draft-tail-host-v271','tree-alternate-chain-target-host-v269','tree-fused-byte-branch-KV-copy-v263','tree-fused-branch-KV-row-copy-v262','tree-static-q5-M120-node-math-v255']:sys.path.append(str(B/f))
sys.path.append(str(B/'tree-contiguous-branch-tail-final-metadata-math-v234'))
for folder in ['tree-static-q5-M128-full463-own-AR-v251','tree-static-q5-M128-projection-host-v249','tree-static-q5-immutable-index-cache-host-v247','tree-static-q5-long-prefix-origins-node-math-v246','tree-static-q5-projection-host-v243','tree-canonical-contiguous-full463-own-AR-v235']:
    sys.path.append(str(B/folder))


def save(folder,name,value):
    encoded=json.dumps(value,indent=2)
    with(Path(folder)/name).open('x')as stream:stream.write(encoded)


def MC2_document(manifest):
    return(dict(profile_sha256=manifest.profile_sha256,digest=manifest.digest,
        routes=[route.consensus_document()for route in manifest.routes])if manifest is not None else None)


def worker(cfg,rank,port,connection,admission_connection=None,compact_worker_result_metrics=False,*,spec,output):
    engine=None
    try:
        from check273 import ready
        ready()
        from diagnostic_authorization import activate_local_diagnostic
        with activate_local_diagnostic(spec):
            import torch
            from vllm_ascend.spec_decode.pearl import native_engine as native
            from extension218 import prepare_install
            entry,_=prepare_install(output)
            # Same target model/metadata/transport/replay hooks, with no 201-case
            # inventory wrapper. This diagnostic never grants production scope.
            from install234 import install_entry
            install_entry(entry,native)
            os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),RANK=str(rank),WORLD_SIZE='4',LOCAL_RANK=str(rank))
            if os.environ.get('HCCL_NPU_SOCKET_PORT_RANGE')=='auto':os.environ.pop('HCCL_NPU_SOCKET_PORT_RANGE')
            engine=native.NativePearlEngine(cfg)
            connection.send(('ready',rank,engine.graph_metrics()))
            completed=False
            while True:
                mode,action,_,_=connection.recv()
                if mode=='exit':break
                if completed or(mode,action)!=('calibrate','genuine-private-alternate-draft-component'):raise RuntimeError('single fixed diagnostic action')
                from startup_receipts_v75 import stage4
                from neural273 import run
                fixture=json.loads((P/'FIXTURE273.json').read_text())
                with torch.inference_mode():
                    report=run(engine,native,lambda label,fn:stage4(engine,label,fn),output,fixture)
                    report['model_dtype']=str(engine.model.embed_tokens.weight.dtype)
                    report['MC2_routes']=MC2_document(engine.mc2_static_route_manifest)
                    save(output,f'rank{rank}-numeric.json',report)
                completed=True
                connection.send(('calibrated',rank,dict(numeric_cases=45,performance=False)))
    except BaseException:
        error=traceback.format_exc()
        try:save(output,f'rank{rank}-last-error.json',dict(rank=rank,error=error,performance=False))
        except BaseException:error+='\nretention failed\n'+traceback.format_exc()
        try:connection.send(('error',error))
        except BaseException:pass
    finally:
        connection.close()
        if admission_connection is not None:admission_connection.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--action',choices=['genuine-private-alternate-draft-component'],required=True);parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    from check273 import ready
    ready()
    from check209 import require_ready
    from base209 import PARENT_ACTION
    original=require_ready();spec=dict(action=PARENT_ACTION,identity=original['parent']['identity'])
    output=DONOR/'actual'/args.tag
    if output.exists():raise RuntimeError('fresh private diagnostic tag')
    output.mkdir(parents=True,exist_ok=False)
    from diagnostic_authorization import activate_local_diagnostic
    with activate_local_diagnostic(spec):
        from startup_config_v132 import make_config
        from vllm_ascend.spec_decode.pearl import api
        api._pearl_worker=partial(worker,spec=spec,output=str(output))
        engine=None
        try:
            engine=api.PEARLEngine(make_config())
            engine._send_all(('calibrate',args.action,None,None))
            replies=engine._receive_all('ROOT273 bounded numeric diagnostics')
            if sorted(replies,key=lambda r:r[1])!=[('calibrated',r,dict(numeric_cases=45,performance=False))for r in range(4)]:raise RuntimeError('all four numeric roles required')
            save(output,'completed.json',dict(replies=replies,performance=False,adopted=False))
        finally:
            from shutdown234 import finish
            exited,exit_wait=finish(engine)
            save(output,'worker-exit-wait.json',exit_wait)
            save(output,'worker-exit.json',exited)


if __name__=='__main__':main()
