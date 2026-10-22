"""Pinned source/dataflow proof for the actual V113 target cleanup scope; no torch import."""
import ast
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'; V=B/'tree-s35-combined-controls-v113'; R=B/'tree-s35-production-v101'
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
S=Path('/root/miniconda3/envs/vllm-hust-dev/lib/python3.11/site-packages')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
 if not v:raise RuntimeError(m)
def anchor(path,*texts):
 s=path.read_text()
 for text in texts:require(text in s,f'actual source anchor absent: {path}: {text}')
 return dict(path=str(path),sha256=sha(path),anchors=list(texts))
census=D/'CPU_FACTORY_SOURCE_CENSUS_NATIVE_V101_20261016.json'
require(sha(census)=='3931b7d28b7a4f2dfc6687601cf80613364283bea91d90c149ee0eb10be09be2','census identity')
c=json.loads(census.read_text())
require(all(sha(p)==h for p,h in c['source_hashes'].items()),'prior census source drift')
spec=importlib.util.spec_from_file_location('_census',D/'audit_cpu_factory_sources_host.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
parent_paths=[Path(p) for p in c['source_hashes']]
new_calls=[];aliases=[];escaped=[]
for p in sorted(V.glob('*.py')):
 if p.name.startswith('test_'):continue
 calls,a,e=mod.scan(p);new_calls+=calls;aliases+=a;escaped+=e
require(not aliases and not escaped and not any(r['expanded_kwargs'] for r in new_calls),'unreviewed alias/escape/kwargs')
implicit=[r for r in new_calls if r['default_device_applies']]
expected={(Path(r['path']).name,r['line'],r['call_source']) for r in c['calls'] if Path(r['path']).parent==R and r['default_device_applies']}
require({(Path(r['path']).name,r['line'],r['call_source']) for r in implicit}==expected,'V113 implicit factory site delta')
# New control/worker code contains no data-inheriting factory invocation.
for p in (V/'combined_runtime_v113.py',V/'combined_controls_v113.py',V/'cpu_default_device_cleanup_host.py',V/'candidate_budget_policy.py'):
 require(not mod.scan(p)[0],f'new actual target code needs dataflow review: {p}')
copy_paths=[]
for p in parent_paths:
 if p.parent!=R:continue
 copy=V/p.name
 if copy.exists() and p.name!='diagnostic_authorization.py':
  require(p.read_bytes()==copy.read_bytes(),f'copied helper drift {p.name}');copy_paths.append(copy)
# Every remaining copied authorization change has no factory site.
require(not mod.scan(V/'diagnostic_authorization.py')[0],'authorization introduced factory')
proof=[]
proof.append(anchor(R/'flexible_scope.py',"if type(value) is not int or value < lo"))
proof.append(anchor(R/'flexible_layout.py',"integer(self.prefix_len,'real prompt cursor',0,self.max_model_len-1)","pos = torch.tensor([self.prefix_len],dtype=torch.int32)"))
proof.append(anchor(R/'host_target_metadata.py','tensor.dtype not in (torch.int32, torch.int64)','return tensor.tolist()','logical_positions.extend(logical)','physical_positions.extend(physical)','type(position) is not int','device = model.embed_tokens.weight.device'))
proof.append(anchor(R/'host_tree_transport.py','round(target_compute_ms*1000.0)','output.token_ids.to(device=\'cpu\',dtype=torch.long)'))
proof.append(anchor(R/'runtime_tree_decode.py','target_started = time.perf_counter()','target_ended = time.perf_counter()','_dt_tree_early_target_record'))
proof.append(anchor(N/'native_engine.py','positions.extend(range(cached_tokens, len(prompt)))','positions.extend(range(chunk.start, chunk.end))','input_token_ids.extend(int(token_id) for token_id in prompt[chunk.start : chunk.end])'))
proof.append(anchor(N/'native_model.py','context_lens=torch.tensor([position + 1 for position in positions], dtype=torch.int32)','device = self.embed_tokens.weight.device'))
proof.append(anchor(R/'poisson_coverage_probe.py',"tokenize=False, add_generation_prompt=True)","all(type(token) is int and token >= 0 for token in row)"))
proof.append(anchor(V/'combined_runtime_v113.py',"report['qualification_control_state'] = stage('qualification-original-control',controls.observe)","qualify_structures(engine,stage_all4=stage)"))
proof.append(anchor(V/'combined_controls_v113.py',"and not self.is_draft"))
tok_paths=[S/'transformers/tokenization_utils_base.py',S/'transformers/tokenization_utils_tokenizers.py',S/'transformers/models/qwen2/tokenization_qwen2.py',S/'transformers/models/auto/tokenization_auto.py']
base=tok_paths[0].read_text();backend=tok_paths[1].read_text()
require('if tensor_type is None:' in base and 'return self' in base,'tokenizer tensor_type None short circuit')
require('tensor_type=return_tensors' in backend,'tokenizer return_tensors binding')
tokcfg=Path('/root/data/shared-models/Qwen3-0.6B/tokenizer_config.json')
require(json.loads(tokcfg.read_text()).get('tokenizer_class')=='Qwen2Tokenizer','actual tokenizer class')
control=D/'cpu_default_device_cleanup_host.py'
require(sha(control)=='5c25a9ac7b6f6a4d9b497032bc372f1d3e5429508aa7be2e40e2a8370257336f','prototype bytes')
require((V/'cpu_default_device_cleanup_host.py').read_bytes()==control.read_bytes(),'prototype copy')
paths=set(parent_paths+copy_paths+tok_paths+[tokcfg,control,V/'combined_runtime_v113.py',V/'combined_controls_v113.py',V/'candidate_budget_policy.py',V/'cpu_default_device_cleanup_host.py',V/'diagnostic_authorization.py',S/'torch/__init__.py',S/'torch/utils/_device.py',S/'torch/overrides.py'])
hashes={str(p):sha(p) for p in sorted(paths)}
roles={}
for r in implicit:
 name=Path(r['path']).name
 if name in ('host_target_metadata.py','host_tree_transport.py','flexible_layout.py'):
  role='active_target_dataflow_closed'
 elif name in ('inventory_memory.py','mc2_inventory_contract.py'):
  role='host_int_boolean_scalar_flags'
 elif name in ('independent_history_v75.py','mixed_target_reference_v75.py','normal_draft_actor_v75.py'):
  role='independent_qualification_original_context_only'
 else:role='draft_role_original_context_only'
 roles.setdefault(role,[]).append(dict(file=name,line=r['line'],call_source=r['call_source']))
require(len(roles['active_target_dataflow_closed'])==3 and len(implicit)==21,'reviewed inventory')
report=dict(schema='HOST.V113.actual-target-default-CPU-factory-caller-scope.v1',created_utc=datetime.now(timezone.utc).isoformat(),
 source_dataflow_closed_for_V113_actual_target_pipeline=True,global_factory_equivalence_proved=False,whole_Native_public_API_equivalence=False,
 source_sha256=hashes,source_sha256_after={p:sha(p) for p in hashes},
 basis=dict(census_path=str(census),census_sha256=sha(census),Native_files=33,Native_three_factory_calls=162,Native_implicit_sites=14,V101_superset_calls=91,V113_implicit_sites=21,
 aliases=[],escaped_factory_references=[],expanded_kwargs=[],new_control_factory_calls=0),
 implicit_site_roles=roles,checked_source_anchors=proof,
 active_target_dataflow=[
  dict(site='flexible_layout.RootOnlyPromptPlan.__post_init__',proof='prefix_len passes exact type(value) is int bounded check before tensor([prefix_len]); starts originate checked prompt cursor or scratch zero and Python integer arithmetic'),
  dict(site='host_target_metadata.make_host_tree_attention_metadata',proof='plan positions/cache_positions must be one-dimensional CPU int32/int64; .tolist returns Python ints; checked_host_tree_slots enforces exact int, bounds and unique slots; implicit context_lens data is a new list of integer arithmetic'),
  dict(site='host_tree_transport.verdict',proof='production target timing is perf_counter float subtraction from early_target record, finite/nonnegative check then round to Python int; actual tokens separately use explicit device=cpu'),
  dict(site='native_model.make_attention_metadata context_lens',proof='actual prefill positions are range(cached_tokens,len(prompt)) or range(chunk.start,chunk.end); normal causal path uses checked CPU integer plan.positions.tolist; mixed uses the same host plan validation; no Tensor-valued positions reaches the implicit factory'),
  dict(site='Native remaining implicit sites',proof='census-pinned source derives Python ints/floats by explicit int/float casts, topology/range arithmetic, or CPU .tolist; ordinary Native mixed flag0 path is excluded from actual target service'),
  dict(site='explicit device factories',proof='active target variables resolve from engine.device/model.embed_tokens.weight.device or existing tensor.device, and host branches use literal cpu; as_tensor Native sites explicitly use non-None tensor.device; Python topology factory arguments remain CPU integer lists even when optional device resolves None'),
  dict(site='actual tokenizer',proof='apply_chat_template(tokenize=False) -> encode with omitted return_tensors default None -> backend BatchEncoding tensor_type=None short-circuits convert_to_tensors; production tokenizer wrapper requires nonempty exact Python int token lists')],
 target_scope=dict(cleanup_only_on_target_ranks=True,draft_constructor_and_runtime_context_unchanged=True,independent_21_scope_qualification_original_context_only=True,old_Native_ordinary_mixed_flag=False,source_bound_to_current_helpers=True),
 excluded_claims=['arbitrary external Native API inputs','arbitrary tokenizer return_tensors options','global factory equivalence','draft cleanup','independent 21-scope qualification repeated in combined mode','full-workload AR equality','new device numerical qualification from this HOST source proof'],
 semantic_counterexample=dict(meta_tensor_inheritance_changes=True,evidence=str(D/'CPU_DEFAULT_DEVICE_CLEANUP_HOST_14TEST_20261016.json'),evidence_sha256=sha(D/'CPU_DEFAULT_DEVICE_CLEANUP_HOST_14TEST_20261016.json')),
 launch_obligations=['bind every source_sha256 byte before launch','require actual V110 component audit passed','strict constructor/mode ownership guards at every switch','original21scope once and original/combined each warm+2zero before shared seal','combined actual whole workload remains an observation, not independent AR equality'],
 actual_execution=dict(argv=sys.argv,python=sys.version,no_Torch_import='torch' not in sys.modules,NPU_started=False))
require(report['source_sha256']==report['source_sha256_after'],'source drift during proof')
output=D/'V113_ACTUAL_TARGET_FACTORY_CALLER_SOURCE_SCOPE.json'
with output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(path=str(output),sha256=sha(output),source_files=len(hashes),passed=True,Native_implicit=14,V113_implicit=21,Torch_imported='torch' in sys.modules)))
