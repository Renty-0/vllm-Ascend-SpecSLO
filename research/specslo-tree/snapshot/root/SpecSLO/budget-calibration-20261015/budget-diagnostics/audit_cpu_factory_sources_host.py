"""Read-only AST census for default-CPU cleanup; does not import torch.

This binds source and dataflow obligations. A constructor-state guard alone is
not a proof for arbitrary Tensor-valued arguments or arbitrary public callers.
"""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
NATIVE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
RUNTIME = P.parent/'tree-s35-production-v101'
FACTORIES = {'torch.tensor', 'torch.as_tensor', 'torch.asarray'}
FLOW = {
 ('native_engine.py',16775): ('stable_sequence_lens', 'stable_sequence_lens = tuple(int(value) for value in sequence_lens)', 'Python tuple of int, explicitly materialized in this function'),
 ('native_engine.py',20411): ('padded_context_buckets', 'sequence_lens = tuple(int(value) for value in attention_metadata.sequence_lens)', 'context_buckets/dummy_context_buckets are tuple integer casts or integer bucket arithmetic; padded_context_buckets is a new starred tuple'),
 ('native_engine.py',14158): ('confidence_values', 'confidence_values = [float(value) for value in confidence.detach().cpu().tolist()]', 'Python list of float; no device Tensor is passed as data'),
 ('native_engine.py',18133): ('[position + 1 for position in envelope.positions]', 'if not self._mixed_target_graph_enabled or self.is_draft:', 'Native ordinary mixed/stable flag0 rejects this path in V101/V110; if enabled, caller must separately prove envelope integer positions'),
 ('native_engine.py',18197): ('[position + 1 for position in envelope.positions]', 'if not self._mixed_target_graph_enabled or self.is_draft:', 'Native ordinary mixed flag0 rejects this path in V101/V110; if enabled, caller must separately prove envelope integer positions'),
 ('native_engine.py',13779): ('[position + 1 for position in physical_positions]', 'physical_positions: list[int] = []', 'New Python list from tree cache-position integer materialization; V110 draft role remains original CPU context'),
 ('native_model.py',2382): ('[position + 1 for position in positions]', 'positions: list[int]', 'New Python list; actual clean-scope callers must bind Python integer positions (fresh-prefix range/int lists or checked CPU positions.tolist), not arbitrary Native API Tensor arguments'),
 ('native_model.py',2526): ('[int(value) + 1 for value in physical_positions]', '[int(value) + 1 for value in physical_positions]', 'New Python list with explicit int conversion for every element'),
 ('native_model.py',2611): ('[int(value) + 1 for value in physical_positions]', '[int(value) + 1 for value in physical_positions]', 'New Python list with explicit int conversion for every element'),
 ('tree.py',411): ('parents_tuple', 'parents_tuple, _, _ = _spine_first_topology(width, depth)', 'Topology helper constructs immutable Python tuple from -1/range integers; cached CPU caller casts geometry arguments to int'),
 ('tree.py',412): ('[0] + _node_levels(width, depth)', '[0] + _node_levels(width, depth)', 'Python list built from literal zero and range-derived integer node levels'),
 ('tree.py',344): ('positions', 'positions=torch.tensor(positions, dtype=position_dtype)', 'Caller passes tuple(int(value) for value in plan.positions.tolist()); cache input is Python integer tuple'),
 ('tree.py',95): ('visible_rows', 'visible_rows.append(node + 1)', 'Python list append from range index arithmetic'),
 ('tree.py',96): ('visible_columns', 'visible_columns.append(root_column + ancestor + 1)', 'Python list append from int(max_model_len), range index and integer topology parents'),
}


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scan(path):
    source = path.read_text(); module = ast.parse(source)
    parents = {child: node for node in ast.walk(module) for child in ast.iter_child_nodes(node)}
    rows, aliases, escaped = [], [], []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            aliases.extend(dict(line=node.lineno,source=ast.unparse(node)) for name in node.names
                           if name.name=='torch' and name.asname not in (None,'torch'))
        elif isinstance(node, ast.ImportFrom) and node.module=='torch':
            aliases.extend(dict(line=node.lineno,source=ast.unparse(node)) for name in node.names
                           if name.name in ('tensor','as_tensor','asarray'))
        if isinstance(node, ast.Attribute) and ast.unparse(node) in FACTORIES:
            parent = parents.get(node)
            if not isinstance(parent, ast.Call) or parent.func is not node:
                escaped.append(dict(line=node.lineno,source=ast.unparse(parent)))
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or ast.unparse(node.func) not in FACTORIES:
            continue
        function = parents.get(node)
        while function is not None and not isinstance(function,(ast.FunctionDef,ast.AsyncFunctionDef)):
            function = parents.get(function)
        device = next((kw.value for kw in node.keywords if kw.arg=='device'), None)
        omitted = device is None or isinstance(device,ast.Constant) and device.value is None
        row = dict(path=str(path),line=node.lineno,function=function.name if function else None,
                   factory=ast.unparse(node.func),call_source=ast.get_source_segment(source,node),
                   data=ast.unparse(node.args[0]) if node.args else None,
                   device_expression=ast.unparse(device) if device is not None else None,
                   default_device_applies=omitted,expanded_kwargs=any(kw.arg is None for kw in node.keywords))
        if path.parent==NATIVE and omitted:
            expected,anchor,proof = FLOW[(path.name,node.lineno)]
            if row['data']!=expected or anchor not in source:
                raise RuntimeError('Native dataflow anchor changed: '+str(path)+':'+str(node.lineno))
            row.update(dataflow=proof,source_anchor=anchor,
                       actual_caller_integer_contract_required=(path.name=='native_model.py' and node.lineno==2382),
                       arbitrary_public_arguments_equivalent=False)
        rows.append(row)
    return rows,aliases,escaped


def main():
    if len(sys.argv)!=2: raise SystemExit('usage: audit_cpu_factory_sources_host.py OUTPUT.json')
    output=Path(sys.argv[1]); output.resolve().parent.mkdir(parents=True,exist_ok=True)
    paths=sorted(NATIVE.glob('*.py'))+sorted(p for p in RUNTIME.glob('*.py') if not p.name.startswith('test_'))
    hashes={str(path):digest(path) for path in paths}
    calls=[]; aliases=[]; escaped=[]
    for path in paths:
        c,a,e=scan(path); calls.extend(c)
        aliases.extend(dict(path=str(path),**row) for row in a)
        escaped.extend(dict(path=str(path),**row) for row in e)
    native=[row for row in calls if Path(row['path']).parent==NATIVE]
    implicit=[row for row in native if row['default_device_applies']]
    if {(Path(r['path']).name,r['line']) for r in implicit}!=set(FLOW):
        raise RuntimeError('Native implicit factory inventory changed')
    n_alias=[row for row in aliases if Path(row['path']).parent==NATIVE]
    n_escape=[row for row in escaped if Path(row['path']).parent==NATIVE]
    if n_alias or n_escape or any(row['expanded_kwargs'] for row in native):
        raise RuntimeError('Native alias/escaped/**kwargs factory requires new review')
    if hashes!={str(path):digest(path) for path in paths}:raise RuntimeError('Source changed during census')
    report=dict(schema='HOST.default-CPU-three-factory-source-census.v1',
        source_hashes=hashes,source_hashes_after=hashes,
        auditor_sha256=digest(__file__),
        Native=dict(files=sum(path.parent==NATIVE for path in paths),calls=len(native),
                    implicit=len(implicit),aliases=n_alias,escaped_factory_references=n_escape,
                    expanded_kwargs=[],device_expression_counts=dict(Counter(row['device_expression'] for row in native))),
        V101_runtime_superset=dict(files=sum(path.parent==RUNTIME for path in paths),
                    calls=sum(Path(row['path']).parent==RUNTIME for row in calls),
                    implicit=sum(Path(row['path']).parent==RUNTIME and row['default_device_applies'] for row in calls)),
        calls=calls,aliases=aliases,escaped_factory_references=escaped,
        passed_Native_source_inventory_and_anchors=True,
        actual_component_caller_proof_required=True,whole_Native_public_API_equivalence=False,
        full_V101_runtime_dataflow_proved=False,
        limits=['Three data-inheriting factories only; arbitrary factories/callers are not universally equivalent',
                'Explicit device variables still require actual caller non-None device/data contracts',
                'Native model make_attention_metadata Python-int positions require the actual component caller proof',
                'V101 copied non-test runtime sources are a census superset, not an executed-call claim',
                'No sys.setprofile, tensor operation, Torch import, model or NPU execution'],
        Torch_imported='torch' in sys.modules,NPU_started=False)
    with output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(output=str(output),sha256=digest(output),Native_calls=len(native),
        Native_implicit=len(implicit),Native_aliases=len(n_alias),Native_escaped=len(n_escape),
        V101_implicit=report['V101_runtime_superset']['implicit'],Torch_imported=False)))


if __name__=='__main__':main()
