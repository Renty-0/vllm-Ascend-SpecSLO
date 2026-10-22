"""Exact startup numerical exports and actual Native bucket parser, no model."""
from pathlib import Path
import ast,hashlib,json,shlex
P=Path(__file__).resolve().parent;B=P.parent
REF=B/'tree-M120-single-graph-prefill-isolation-v448/startup448.sh'
BAD=B/'tree-M120-single-graph-prefill-scope-v449/startup449.sh'
def settings(path):
    out={}
    for line in path.read_text().splitlines():
        if line.startswith('export '):
            for token in shlex.split(line[7:]):
                if '='in token:
                    k,v=token.split('=',1)
                    if k.startswith('VLLM_ASCEND_')or k in('HCCL_DETERMINISTIC','HCCL_OP_EXPANSION_MODE','TASK_QUEUE_ENABLE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS'):out[k]=v
    return out
actual=settings(P/'startup451.sh');valid=settings(REF);bad=settings(BAD)
assert actual==valid,'All numerical/runtime settings must match the known valid448 startup'
file=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py')
text=file.read_text();tree=ast.parse(text)
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='resolve_mixed_target_graph_buckets')
constant=next(n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='MIXED_TARGET_PROMPT_TOKEN_BUCKETS'for t in n.targets))
namespace={'MIXED_TARGET_PROMPT_TOKEN_BUCKETS':ast.literal_eval(constant.value)}
exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(file)+'::exact-host-bucket-parser','exec'),namespace)
key='VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_BUCKETS'
parsed=namespace['resolve_mixed_target_graph_buckets'](actual[key])
assert parsed==(64,128,192,256,320,384,448,512)
try:namespace['resolve_mixed_target_graph_buckets'](bad[key])
except ValueError:pass
else:raise AssertionError('Previously invalid449 bucket must be rejected')
out=dict(status='PASS_HOST451_EXACT_VALID_NUMERICAL_EXPORTS_AND_ACTUAL_NATIVE_PARSER',
    numerical_settings=actual,native_source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
    actual_parser_body_sha256=hashlib.sha256(ast.get_source_segment(text,function).encode()).hexdigest(),
    valid_buckets=list(parsed),previous449_invalid_setting_rejected=True,
    NPU_used=False,model_constructors=0,Raw=None,Goodput=None)
with(P/'HOST451_SETTINGS_RESULT.json').open('x')as stream:json.dump(out,stream,indent=2);stream.write('\n')
print(json.dumps(dict(status=out['status'],valid_buckets=list(parsed),previous449_rejected=True,NPU_used=False)))
