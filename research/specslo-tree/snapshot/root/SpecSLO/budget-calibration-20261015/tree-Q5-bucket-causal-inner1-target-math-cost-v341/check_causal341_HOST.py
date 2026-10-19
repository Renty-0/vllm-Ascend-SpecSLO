"""Exercise exact original FIA callbacks with an operator recorder, no torch."""
import ast
import __future__
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import causal341

P=Path(__file__).resolve().parent
SOURCE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_graph.py')


def main():
    tree=ast.parse(SOURCE.read_text())
    names=('run_native_fused_infer_attention','_update_fused_infer_attention_task','_validate_tree_fia_mask')
    nodes=[copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)and n.name==name))for name in names]
    calls=[]
    op=SimpleNamespace(out=lambda **kwargs:calls.append(kwargs))
    fake_torch=SimpleNamespace(empty=lambda *a,**k:object(),bool='bool',
        npu=SimpleNamespace(graph_task_update_begin=lambda *a:None,graph_task_update_end=lambda *a:None))
    ns=dict(torch=fake_torch,torch_npu=SimpleNamespace(npu_fused_infer_attention_score=op),
        _CAPTURED_TASKS=None,TREE_FIA_SPARSE_MODE=1,TREE_FIA_INNER_PRECISE=1)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec',flags=__future__.annotations.compiler_flag,dont_inherit=True),ns)
    forward,fp=causal341.project(ns[names[0]])
    update,up=causal341.project(ns[names[1]])
    query=SimpleNamespace(dtype='BF16',device='HOST',shape=(120,24,128))
    mask=SimpleNamespace(ndim=4,dtype='bool',shape=(24,1,10,4096))
    table=SimpleNamespace(ndim=2,shape=(24,32))
    fields=dict(query=query,key_cache=object(),value_cache=object(),num_kv_heads=3,
        num_heads=24,scale=128**-.5,block_table=table,attention_mask=mask,
        actual_seq_lengths_q=list(range(5,121,5)),actual_seq_lengths_kv=[2560]*24,
        block_size=128,output=object())
    for full in (False,True):
        ns[names[0]](**fields,tree_attention=full)
        original=calls.pop()
        forward(**fields,tree_attention=full)
        generated=calls.pop()
        assert generated['sparse_mode']==(1 if full else 3) and generated['inner_precise']==1
        original.pop('out');generated.pop('out')
        if not full:
            assert 'inner_precise'not in original
            generated.pop('inner_precise')
        assert generated==original
        task=SimpleNamespace(query=query,key_cache=fields['key_cache'],value_cache=fields['value_cache'],
            attention_mask=mask,block_table=table,block_size=128,num_kv_heads=3,num_heads=24,
            scale=128**-.5,tree_attention=full,handle=object(),workspace=object(),
            output=object(),softmax_lse=object())
        update(SimpleNamespace(),task,fields['actual_seq_lengths_q'],fields['actual_seq_lengths_kv'],stream=object(),record_event=False)
        refreshed=calls.pop()
        assert refreshed['sparse_mode']==(1 if full else 3) and refreshed['inner_precise']==1
    assert forward.__globals__ is update.__globals__ is ns
    marker=[];ns['_CAPTURED_TASKS']=marker
    assert forward.__globals__['_CAPTURED_TASKS']is marker
    result=dict(status='passed_original_FIA_callbacks_HOST',NPU_used=False,
        capture_and_update_causal_arguments_identical=True,FULL_arguments_unchanged=True,
        original_live_capture_globals_preserved=True,forward=fp,update=up,
        fake_operator_not_device_evidence=True)
    (P/'HOST341_CAUSAL_ABI_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
