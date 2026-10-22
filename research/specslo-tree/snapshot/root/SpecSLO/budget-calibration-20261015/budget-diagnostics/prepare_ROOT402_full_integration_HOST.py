"""Prepare, but do not launch or freeze, the normal24/global64 full candidate."""
import hashlib
import json
from pathlib import Path
import re

B = Path('/root/SpecSLO/budget-calibration-20261015')
OLD = B / 'tree-active64-whole-proposal-segment-limit-v398'
P = B / 'tree-global64-normal24-typed-admission-v402'


def rename_text(source):
    parts = re.split('([0-9a-f]{64})', source)
    result = ''.join(part if re.fullmatch('[0-9a-f]{64}', part) else
                     part.replace(str(OLD), str(P)).replace(OLD.name, P.name).replace('398', '402')
                     for part in parts)
    return '\n'.join(line.rstrip() for line in result.splitlines()) + '\n'


def replace_one(path, before, after):
    source = path.read_text()
    assert source.count(before) == 1, (path.name, before)
    path.write_text(source.replace(before, after))


def main():
    typed = P / 'check402_HOST.py'
    target = P / 'check402_typed_HOST.py'
    assert typed.exists() and not target.exists()
    target.write_text(typed.read_text().replace('HOST402_RESULT.json', 'HOST402_TYPED_ALLOCATION_RESULT.json'))
    typed.unlink()
    (P / 'HOST402_RESULT.json').rename(P / 'HOST402_TYPED_ALLOCATION_RESULT.json')
    copied = {}
    for source in sorted(OLD.iterdir()):
        if source.suffix not in {'.py', '.sh'} or source.name.startswith('HOST398.'):
            continue
        destination = P / source.name.replace('398', '402')
        assert not destination.exists(), destination
        destination.write_text(rename_text(source.read_text()))
        destination.chmod(source.stat().st_mode & 0o777)
        copied[str(destination)] = dict(source=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    replace_one(P / 'check402.py', "'HOST402_LIVE_BINDING_RESULT.json'):",
                "'HOST402_LIVE_BINDING_RESULT.json', 'HOST402_TYPED_ALLOCATION_RESULT.json'):")
    replace_one(P / 'extension402.py', 'original_best257_draft_allocator_and_neural_math_preserved=True,',
                'original_best257_draft_allocator_and_neural_math_preserved=False,\n'
                '        original_best257_draft_and_neural_math_preserved=True, actual_normal_draft_policy_limit=24,')
    replace_one(P / 'runtime402.py', "report['ROOT402_uniform_KV_math']=extension.bucket_snapshot(engine)\\n",
                "report['ROOT402_uniform_KV_math']=extension.bucket_snapshot(engine)\\n"
                "     report['ROOT402_normal_admission']=__import__('shape402').snapshot(engine)\\n")
    replace_one(P / 'audit402.py', 'ROOT402_original_best257_draft_and_allocator_unchanged=True)',
                'ROOT402_original_best257_draft_and_allocator_unchanged=False,\n'
                '        ROOT402_best257_draft_neural_math_unchanged=True, ROOT402_normal_work_policy_limit=24)')
    replace_one(P / 'guard402.py', 'original_best257_optimized_draft_unchanged=True, active_request_limit=64,',
                'original_best257_optimized_draft_unchanged=True, normal_work_admission_policy_limit=24,\n'
                '                   original_gain_model_and_typed_role_limits_unchanged=True, active_request_limit=64,')
    replace_one(P / 'generation402.py', '    function.body.insert(index, imported)\n',
                "    function.body.insert(index, imported)\n"
                "    imported_shape = ast.parse('from shape402 import shape as _shape402').body[0]\n"
                "    function.body.insert(index, imported_shape)\n"
                "    shaped = [node for node in ast.walk(tree) if isinstance(node, ast.Call)\n"
                "              and ast.unparse(node.func) == 'shaper.shape']\n"
                "    if len(shaped) != 1:\n"
                "        raise RuntimeError('ROOT402 one actual typed shaper call')\n"
                "    shaped_call = shaped[0]\n"
                "    shaped_original = copy.deepcopy((shaped_call.func, shaped_call.args))\n"
                "    shaped_call.func = ast.Name(id='_shape402', ctx=ast.Load())\n"
                "    shaped_call.args[:0] = [ast.Name(id='self', ctx=ast.Load()), ast.Name(id='shaper', ctx=ast.Load())]\n")
    replace_one(P / 'generation402.py', '    function.body.remove(imported)\n',
                '    function.body.remove(imported)\n'
                '    function.body.remove(imported_shape)\n'
                '    shaped_call.func, shaped_call.args = shaped_original\n')
    replace_one(P / 'generation402.py', 'explicit_segment_packing402=packing_descriptor(), actual_global_active_limit=64)',
                'explicit_segment_packing402=packing_descriptor(), actual_global_active_limit=64,\n'
                '                actual_typed_normal_work_admission_limit=24, original_gain_model_unchanged=True)')
    report = dict(status='PREPARED_UNFROZEN_NOT_LAUNCHED', cloned=copied,
                  baseline398_source_freeze_sha256='dfb6cfd44d2f524d7ad28892ec055d50d1035910fb71dbbc5486dc060cb5e73d',
                  global_active_limit=64, target_limit=24, normal_work_policy_limit=24,
                  draft_neural_math_changed=False, typed_gain_model_changed=False,
                  normal_admission_changed=True, model_qualification=False,
                  pending=['extended normal admission terminal audit', 'all actual HOST startup/source bindings',
                           'fresh freeze/archive', 'new full463 independent own-AR and actual metrics'],
                  NPU_launched=False, Raw=None, Goodput=None)
    (P / 'DERIVATION402.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], cloned_files=len(copied))))


if __name__ == '__main__':
    main()
