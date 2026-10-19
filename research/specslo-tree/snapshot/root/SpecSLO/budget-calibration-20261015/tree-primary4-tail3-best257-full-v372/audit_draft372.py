"""Re-use original371 mathematical auditor; explicitly audit qualification-only scope."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
P371 = BASE / 'tree-primary4-tail3-eager-allN-device-v371'


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT372 fresh draft audit: ' + message)


def neural_body_audit(directory):
    if str(P371) not in sys.path:
        sys.path.append(str(P371))
    import audit371
    source = (P371 / 'audit371.py').read_text()
    original = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'audit')
    node = copy.deepcopy(original)
    # The prefix includes the entire original306-case mathematical audit.
    cuts = [i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
            and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'costs']
    need(len(cuts) == 1, 'one original mathematical/cost audit boundary')
    boundary = cuts[0]
    old_suffix = copy.deepcopy(node.body[boundary:])
    node.body[boundary:] = ast.parse('''
need(a['costs'] == [] and a['median_seconds'] == {}, 'qualification-only: no measured or inherited component costs')
need(a['all_committed_prefix_KV_protected'] is True and a['native_qualification_bodies_unchanged'] is True,
     'all committed prefixes protected and original Native checks')
need(a['all_logical_N_1_to32_exercised'] is True and a['normal_mixed_and_all_eager_exercised'] is True,
     'all original logical shapes and eager contracts')
need(a['Raw'] is None and a['Goodput'] is None and a['adopted'] is False and a['serving_installed'] is False,
     'no standalone performance or LIVE adoption claimed')
return a
''').body
    projected = copy.deepcopy(node)
    node.body[boundary:] = old_suffix
    need(ast.dump(node, include_attributes=False) == ast.dump(original, include_attributes=False),
         'entire original mathematical-auditor AST inverse')
    namespace = dict(vars(audit371))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[projected], type_ignores=[])),
                 __file__ + '::unchanged-original371-math-prefix', 'exec'), namespace)
    path = Path(directory) / 'rank0-original371-neural-body.json'
    # Keep the original file read expression and use a private path wrapper.
    class Folder:
        def __truediv__(self, child):
            need(child == 'result.json', 'one original result-file read')
            return path
    # Original audit calls Path(directory); accepting a Path alias is the only
    # file-name projection, and does not change parsed data or math assertions.
    namespace['Path'] = lambda value: Folder()
    return namespace['audit'](directory)


def audit(directory):
    directory = Path(directory)
    a = neural_body_audit(directory)
    proof_path = directory / 'rank0-tail3-same-instance-proof.json'
    proof = json.loads(proof_path.read_text())
    need(proof['cases'] == 306 and proof['fresh_same_model_instance'] is True
         and proof['private_entries'] == 64 and proof['all_new_private_entries_naturally_qualified'] is True,
         'fresh same-engine306 math and all64 naturally qualified private graphs')
    need(proof['costs_measured'] is False and proof['native_qualification_flag_assignments'] is False
         and proof['original_warm_twozero_seal_required_for_all_entries'] is True,
         'no inherited costs or Native-flag assignment; original global warm gates retained')
    original = directory / 'rank0-original371-neural-body.json'
    need(proof['original_body_result_sha256'] == hashlib.sha256(original.read_bytes()).hexdigest(),
         'exact original neural result bytes')
    return dict(passed=True, cases=306, new_private_graph_entries=64,
                all_original371_math_assertions_retained=True,
                original_body_result_sha256=proof['original_body_result_sha256'],
                proof_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),
                component_costs_measured=False, whole463_qualified=False)
