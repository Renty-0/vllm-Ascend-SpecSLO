"""Real Torch CPU + original Native HOST metadata/cache ABI, no model math.

Only the graph runner is a stub. No torch_npu import, NPU, attention kernel,
weights, neural output, or performance qualification is involved.
"""
import ast
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any

B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
P=B/'tree-packed-KV-preflight-rank-aware-own-AR-full-v218'
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')


def need(value,message):
    if not value:raise AssertionError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch
    need('torch_npu'not in sys.modules,'controlled CPU-only backend import')
    spec=importlib.util.spec_from_file_location('_ROOT218_actual_native_cache_CPU',N/'native_cache.py')
    cache=importlib.util.module_from_spec(spec);sys.modules[spec.name]=cache;spec.loader.exec_module(cache)
    namespace=dict(torch=torch,dataclass=dataclass,Any=Any,NativeCacheAllocation=cache.NativeCacheAllocation)
    source=(N/'native_model.py').read_text();tree=ast.parse(source)
    metadata=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='NativeAttentionMetadata'))
    method=copy.deepcopy(next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='make_attention_metadata'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[metadata,method],type_ignores=[])),str(N/'native_model.py')+'::unaltered-CPU-HOST-metadata','exec'),namespace)
    tree=ast.parse((N/'native_engine.py').read_text())
    names={'_allocate_cache','_ensure_cache_capacity','_cache_slot_mapping','_prepare_attention_metadata'}
    methods=[copy.deepcopy(n)for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name in names]
    need({n.name for n in methods}==names,'four original Native HOST cache/metadata methods')
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),str(N/'native_engine.py')+'::unaltered-CPU-HOST-cache','exec'),namespace)
    sys.path[:0]=[str(P),str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209')];import base209
    import reference218
    from mapper173 import GREEDY
    original=B/'tree-causal80-small-cohort-role-aware-audit-full-v209/actual/cache-slo1-root-v213-full463-v1/windows/window32/full463/rank1-full463-actual-trace.json.gz'
    prompts=json.loads(gzip.decompress(original.read_bytes()))['tokenized_inputs']
    records=[]
    class GraphStub:
        graph_cache_sealed=True
        def __init__(self):
            self.target_entries={GREEDY:types.SimpleNamespace(runtime_validated=True)}
            self.disabled_entry_keys=set()
        def run_target_greedy(self,inputs,positions,metadata,vocab):
            need(len(inputs)==len(positions)==len(metadata)==1,'original actual target-greedy ABI')
            need(inputs[0].shape==positions[0].shape==(80,)and vocab==151936,'original Q80 input/position geometry')
            return [torch.arange(80,dtype=torch.long,device='cpu')]
    for n in(1,2,8,16,24,32):
        model=types.SimpleNamespace(embed_tokens=types.SimpleNamespace(weight=torch.zeros(1,device='cpu')),
            layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(block_table=torch.zeros((527,32),dtype=torch.int32,device='cpu'),blocks_per_sequence=32,block_size=128))],
            attention_mask=torch.zeros((2048,2048),dtype=torch.int8,device='cpu'))
        model.make_attention_metadata=types.MethodType(namespace['make_attention_metadata'],model)
        engine=types.SimpleNamespace(device=torch.device('cpu'),prefix_cache=cache.NativePrefixCache(544,32,128),
            _cache_storage_sequence_capacity=527,config=types.SimpleNamespace(kvcache_block_size=128),model=model,
            graph_runner=GraphStub(),draft_vocab_size=151936)
        for name in names:setattr(engine,name,types.MethodType(namespace[name],engine))
        source=[list(p)for p in prompts[:n]]
        engine._allocate_cache(source,enable_prefix_caching=False,reserve_sequence_capacity=True)
        for begin in range(0,n,16):
            selected=list(range(begin,min(begin+16,n)))
            for offset in(0,127,254):
                outputs=[[151667]+[198]*offset for _ in range(n)]
                for future in(0,151643):
                    data=reference218.frame(source,outputs,selected,future=future)
                    row=reference218.target_step(engine,data)
                    expected=[engine.cache_allocation.block_tables[i][p//128]*128+p%128 for i,p in zip(data['ids'],data['positions'],strict=True)]
                    need(row['metadata']['slots']==expected and row['metadata']['contexts']==[p+1 for p in data['positions']], 'real original producer slot/context ABI')
                    need(row['metadata']['sequence_lens']==data['sequence_lens']and row['metadata']['actual_seq_lengths_q']==list(range(5,81,5)), 'real original FIA request and query lengths')
                    need(row['real_tokens']==list(range(0,5*len(selected),5)), 'actual reference consumes first query of each real owner only')
                    records.append(dict(cohort_owners=n,selected=selected,output_offset=offset,future=future,passed=True))
    need('torch_npu'not in sys.modules,'no accelerator module loaded')
    paths=[Path(__file__),N/'native_engine.py',N/'native_model.py',N/'native_cache.py',P/'reference218.py',original]
    out=D/'ROOT_V218_REFERENCE_REAL_HOST_METADATA_CPU_CONTROL.json'
    need(not out.exists(),'fresh HOST-only evidence')
    result=dict(schema='ROOT218.reference-original-HOST-metadata-CPU.v1',UTC=datetime.now(timezone.utc).isoformat(),status='passed',
        cases=len(records),rows=records,source_sha256={str(p):sha(p)for p in paths},
        original_Native_cache_and_metadata_bodies_used=True,graph_runner_stub=True,
        Torch_imported=True,Torch_npu_imported=False,NPU_used=False,neural_model_computed=False,
        numerical_output_qualified=False,Goodput=None,adopted=False)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in('rows','source_sha256')}))


if __name__=='__main__':main()
