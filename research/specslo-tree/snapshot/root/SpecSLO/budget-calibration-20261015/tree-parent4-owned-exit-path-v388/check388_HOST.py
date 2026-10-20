"""Check the actual exit import in the parent and a real spawned HOST child."""
import json
import multiprocessing
from pathlib import Path
import sys

P=Path(__file__).parent;B=P.parent
sys.path.insert(0,str(P))


def child(connection):
    import shutdown234
    connection.send(str(Path(shutdown234.__file__).resolve()));connection.close()


def main():
    import runtime388,shutdown234,guard388
    expected=B/'tree-contiguous-branch-tail-final-metadata-math-v234/shutdown234.py'
    assert Path(shutdown234.__file__).resolve()==expected
    context=multiprocessing.get_context('spawn');left,right=context.Pipe(duplex=False)
    process=context.Process(target=child,args=(right,));process.start();right.close()
    assert left.poll(30)and left.recv()==str(expected);left.close();process.join(30)
    assert process.exitcode==0
    ns=guard388.execution_namespace()
    assert ns['execute_locked'].__globals__ is ns and ns['prepared'] is guard388.prepared
    assert ns['SCOPE']['real_draft_generation_requested'] is True
    assert str(P/'startup388.sh')in guard388.launch_argv('HOST388-only')
    assert 'torch'not in sys.modules and 'torch_npu'not in sys.modules
    result=dict(status='PASS_HOST388_REAL_SPAWN_EXIT_IMPORT_AND_SUPERVISOR',NPU_used=False,
        actual_parent_and_spawn_exit_import=str(expected),model_constructed=False,
        source_gate_delta_only=True,models_qualified=False,Raw=None,Goodput=None)
    (P/'HOST388_CONSTRUCTION_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
