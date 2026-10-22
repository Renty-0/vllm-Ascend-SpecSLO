"""Bounded source reflection only. No original decoder/model/protocol is executed."""
import ast,copy,hashlib,importlib.abc,importlib.machinery,importlib.util,inspect,json,sys,textwrap,types
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-linear-parent4-child3-bank-binding-full-v179';V148=B/'tree-high-normal-cap4-native-families-v148'
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(value,message):
 if not value:raise RuntimeError(message)
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers'):raise ImportError('HOST backend firewall '+name)
sys.meta_path.insert(0,Firewall());sys.path.insert(0,str(P))
original_local={str(p):sha(p) for p in P.iterdir() if p.is_file()}
import extension179
# Actual prepare does not construct engine or call any returned install function.
entry,controls=extension179.prepare_install(P)
check('native_protocol' not in sys.modules,'unexpected preloaded module invalidates missing-import test')
check(importlib.machinery.PathFinder.find_spec('native_protocol',sys.path) is None,'unexpected path supplies native_protocol')
try:__import__('native_protocol')
except ModuleNotFoundError as error:missing=dict(type=type(error).__name__,message=str(error))
else:raise RuntimeError('missing-module negative failed')
pathproof=dict(actual_prepare_install_called=True,returned_entry_install_not_called=True,
 native_protocol_loaded=False,native_protocol_found_by_actual_sys_path=False,error=missing,sys_path=list(sys.path),Torch_imported=False)
# A namespace carrier provides ONLY the import-time decorator and module symbols.
# It does not implement Tensor, model, collective or device operations.
torch=types.ModuleType('torch');torch.__path__=[];torch.HOST_EXPLICIT_IMPORT_CARRIER=True
torch.inference_mode=lambda:lambda fn:fn
dist=types.ModuleType('torch.distributed');torch.distributed=dist
sys.modules['torch']=torch;sys.modules['torch.distributed']=dist
native=types.ModuleType('_HOST_original_Native_source_only')
namespace=vars(native);namespace.update(torch=torch,__file__=str(NATIVE))
tree=ast.parse(NATIVE.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NativePearlEngine')
fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_generate_spec_rhythm_tree_decode'))
future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
exec(compile(ast.fix_missing_locations(ast.Module(body=[future,fn],type_ignores=[])),str(NATIVE),'exec'),namespace)
native.NativePearlEngine=type('HOSTSourceOnlyNativePearlEngine',(),{'_generate_spec_rhythm_tree_decode':namespace['_generate_spec_rhythm_tree_decode']})
from original_tree_cycle_v75 import capture_original_protocol_before_startup,_require_compiled
captured=capture_original_protocol_before_startup(native);_require_compiled(captured)
pre=dict(original_decode_sha256=captured['original_decode_sha256'],canonical_decode_sha256=captured['canonical_decode_sha256'],
 functions={k:dict(callable=callable(captured[k]),code_filename=captured[k].__code__.co_filename) for k in ('preflight_consensus_compact','finish','cleanup')},
 actual_commit_consensus_observations=captured['actual_commit_consensus_observations'],source_only_functions_not_executed=True)
# Execute the exact fresh_decode DEFINITION, preserving its real source location;
# do not call it, which would perform actual inference/progress work.
entrypath=V148/'tree_entry.py';entrytree=ast.parse(entrypath.read_text())
fresh=copy.deepcopy(next(n for n in ast.walk(entrytree) if isinstance(n,ast.FunctionDef) and n.name=='fresh_decode'))
freshns={'__name__':'_HOST_actual_fresh_decode_definition'}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fresh],type_ignores=[])),str(entrypath),'exec'),freshns)
native.NativePearlEngine._generate_spec_rhythm_tree_decode=freshns['fresh_decode']
freshsource=textwrap.dedent(inspect.getsource(native.NativePearlEngine._generate_spec_rhythm_tree_decode))
check('preflight_error' not in freshsource and 'decode_impl(engine,**kwargs)' in freshsource,'actual fresh wrapper reflection')
protocolpath=B/'tree-packed-eager01-cost-v102/native_protocol.py'
spec=importlib.util.spec_from_file_location('_HOST_unchanged_P102_native_protocol',protocolpath);protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
try:protocol.compile_protocol(native)
except BaseException as error:late=dict(type=type(error).__name__,message=str(error),args_repr=repr(error.args))
else:raise RuntimeError('late compile unexpectedly accepted fresh wrapper')
try:capture_original_protocol_before_startup(native)
except BaseException as error:late_capture=dict(type=type(error).__name__,message=str(error))
else:raise RuntimeError('strict original capture did not reject already-installed wrapper')
# Reusing the already captured original object remains valid after installation.
_require_compiled(captured)
check(sha(protocolpath)==sha(V148/'original_native_protocol_v75.py'),'commit_states provider source identity')
check(original_local=={str(p):sha(p) for p in P.iterdir() if p.is_file()},'frozen179 source changed')
pins=[P/'runtime179.py',P/'extension179.py',P/'base179.py',P/'actor179.py',P/'protocol179.py',P/'binding179.py',P/'live179.py',
 B/'tree-live-shared4-union-full-v168/natural168.py',B/'tree-natural-positive-union-discovery-local-v154/actor154.py',
 V148/'tree_entry.py',V148/'original_tree_cycle_v75.py',V148/'original_native_protocol_v75.py',V148/'canonical_eager.py',V148/'host_tree_transport.py',protocolpath,NATIVE,Path(__file__)]
row=dict(schema='queued168-179.protocol-reflection.actual-HOST.v1',passed=True,actual_prepare=pathproof,
 actual_original_prestartup_capture=pre,actual_fresh_decode_reflection=dict(source_sha256=hashlib.sha256(freshsource.encode()).hexdigest(),contains_original_preflight=False,never_called=True),
 actual_late_compile_failure=late,actual_late_strict_capture_failure=late_capture,
 retained_original_object_still_passes_original_require_compiled=True,commit_states_provider_bytes_equal=True,
 conclusion='Frozen179 must remain disabled: actual import is missing before natural warm; a path-only fix still fails late compile on fresh_decode. Frozen168 inherits same actor154 late compiler and missing canonical import. New181 must explicitly bind commit_states provider and reuse current engine original prestartup compiled identity.',
 no_original_decoder_execution=True,no_compiled_protocol_execution=True,no_model_or_Tensor_implementation=True,Torch_library_imported=False,NPU_used=False,current180_raw_read=False,frozen_sources_unchanged=True,
 source_pins={str(p):sha(p) for p in pins})
p=D/'QUEUED168_179_NATIVE_PROTOCOL_ACTUAL_HOST_REFLECTION_EVIDENCE.json'
with p.open('x') as stream:json.dump(row,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(dict(path=str(p),sha256=sha(p),missing=missing,late_compile=late,late_strict_capture=late_capture)))
