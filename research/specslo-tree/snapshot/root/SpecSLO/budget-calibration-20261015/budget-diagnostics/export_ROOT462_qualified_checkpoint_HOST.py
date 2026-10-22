"""Export exact tested source bytes and small evidence to the user's branch."""

from pathlib import Path
import ast
import datetime
import gzip
import hashlib
import json
import shutil

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
REPO = Path('/root/SpecSLO/checkpoints/github-SpecSLO-Tree')
ROOT = REPO / 'research/specslo-tree'
DEST = ROOT / 'adaptive-M80-M120-full-v462'
CLOSURE = D / 'ROOT462_COMPLETE_PREPARED_SOURCE_CLOSURE_HOST.json'
CLOSURE_SHA = 'bac7fcdbfab6b8d7a23d7e8384481abc3bf219e3e937a7f68cf308c16fbb8db9'
TAG = 'adaptive-own-history-global64-root-v462-qualified-wait-full463-ownAR-v2'
RECEIPT = B / 'tree-causal80-small-cohort-role-aware-audit-full-v209' / f'receipt-{TAG}.json'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    closure_bytes = CLOSURE.read_bytes()
    assert sha(closure_bytes) == CLOSURE_SHA
    closure = json.loads(closure_bytes)
    result = json.loads((D / 'ROOT462_V2_FINAL_COMMON_CLOCK_HOST.json').read_text())
    receipt = json.loads(RECEIPT.read_text())
    assert receipt['status'] == 'complete' and receipt['sources_unchanged'] is True
    assert receipt['devices_after']['all_idle'] is True and not receipt['owned_remaining']
    assert result['metrics']['all']['Goodput'] > 439.1622144545149
    assert result['goal_achieved'] is False
    assert not DEST.exists()
    DEST.mkdir(parents=True)
    manifest_path = ROOT / 'SOURCE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    saved = {row['saved']: row for row in manifest['files']}
    omitted = manifest['omitted_evidence']
    source_roots = (Path('/root/SpecSLO'), Path('/root/data/specslo-baselines'),
                    Path('/root/data/vllm-ascend-hust'), Path('/root/data/vllm-hust'))
    text_suffixes = {'.py', '.sh', '.json', '.md', '.toml', '.yaml', '.yml',
                     '.cfg', '.ini', '.cpp', '.cc', '.c', '.h', '.hpp', '.cmake'}
    copied = []

    def snapshot(source, wanted=None):
        source = Path(source)
        data = source.read_bytes()
        digest = sha(data)
        if wanted is not None:
            assert digest == wanted, str(source)
        destination = ROOT / 'snapshot' / source.relative_to('/')
        assert not destination.is_symlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        assert sha(destination.read_bytes()) == digest
        if source.suffix == '.py':
            ast.parse(data, filename=str(source))
        key = str(destination.relative_to(REPO))
        saved[key] = dict(source=str(source), saved=key, bytes=len(data), sha256=digest)
        omitted.pop(str(source), None)
        copied.append(key)

    for name, wanted in closure['files'].items():
        source = Path(name)
        allowed = any(source.is_relative_to(root) for root in source_roots)
        excluded = any(part in ('actual', 'HOST-attempts', 'private-evidence-archives',
                                'shared-models', 'shared-datasets', '__pycache__', '.git')
                       for part in source.parts)
        allowed = allowed and not excluded and source.suffix in text_suffixes
        allowed = allowed and source.stat().st_size <= 2 * 1024**2
        allowed = allowed and not source.name.lower().startswith(('fixture', 'prefixes-data'))
        if allowed:
            snapshot(source, wanted)
        else:
            omitted[name] = dict(sha256=wanted, reason='Historical evidence, data fixture, archive, binary or large artifact; regenerate under original conditions')

    # Include the current independent readers, whose execution is outside the
    # model worker's frozen source graph. No raw tensors or request outputs.
    for source in sorted(D.glob('*.py')):
        if source.stat().st_size < 512 * 1024:
            snapshot(source)
    for source in (CLOSURE, Path('/root/SpecSLO/tpot-report-20261015/DATASET_TPOT_AND_OPTIMIZATION_REPORT_zh.md')):
        snapshot(source)

    evidence = {}
    small_names = (
        'ROOT462_V2_FINAL_COMMON_CLOCK_HOST.json',
        'ROOT462_V2_ADDITIONAL693_INDEPENDENT_NN_READER_HOST.json',
        'ROOT462_V2_ACTUAL_PREFILL_PAUSE_COMPONENT_HOST.json',
        'ROOT462_V2_PROSPECTIVE_PREFILL_COST_VALIDATION_HOST.json',
        'ROOT462_PROSPECTIVE_PREFILL_COST_MODEL_HOST.json',
        'ROOT462_V2_SECOND_LIFECYCLE_PASS_COUNTER_AUDIT_HOST.json',
    )
    for name in small_names:
        source = D / name
        snapshot(source)
        target = DEST / 'evidence' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        evidence[str(target.relative_to(DEST))] = dict(source=str(source), sha256=sha(target.read_bytes()))
    for source, name in (
        (D / 'ROOT462_V2_TERMINAL_FULL_AUDIT_HOST.json', 'terminal-independent-audit.json.gz'),
        (RECEIPT, 'terminal-supervisor-receipt.json.gz'),
    ):
        original = source.read_bytes()
        packed = gzip.compress(original, mtime=0)
        target = DEST / 'evidence' / name
        target.write_bytes(packed)
        assert gzip.decompress(packed) == original
        evidence[str(target.relative_to(DEST))] = dict(source=str(source), sha256=sha(packed),
                                                     uncompressed_sha256=sha(original), compression='gzip')

    validation = dict(
        status='FULL462_GLOBAL64_COMPLETE_INDEPENDENTLY_AUDITED_SINGLE_RUN_GAIN',
        global_active_limit=64, actual_active_peak=48, Raw=result['metrics']['all']['Raw'],
        Goodput=result['metrics']['all']['Goodput'], attained=313, requests=463,
        previous_global64_best_Goodput=439.1622144545149,
        Goodput_ratio=result['GP_over_previous_global64_best'],
        target_ranks=3, exact_requests_each_target=463, exact_tokens_each_target=118528,
        original_node_cases=227, original_uniform_KV_cases=681, additional_own_physical_cases=693,
        lifecycle_cases=75, incumbent_prefill_cases=6, source_and_environment_checks_passed=True,
        source_freeze_SHA256=result['source_freeze_SHA256'], source_archive_SHA256=result['source_archive_SHA256'],
        final_matched_repeats=0, performance_goal_achieved=False,
        format_ci='Attempted; pre-commit is not installed. No environment install or upgrade performed.',
        snapshot_has_NPU_qualification=False, model_or_dataset_copied=False,
        raw_NPU_tensors_or_graph_caches_copied=False, original_byte_exact_full_backup=False,
    )
    (DEST / 'VALIDATION.json').write_text(json.dumps(validation, indent=2) + '\n')
    (DEST / 'EXPORT_MANIFEST.json').write_text(json.dumps(evidence, indent=2) + '\n')
    (DEST / 'README_zh.md').write_text(
        '# v462：已验证的自适应M80/M120完整版本\n\n'
        '单次完整463请求与三路独立own-AR均通过；全局上限64，实际峰值48。'
        'Raw681.6589、Goodput460.8191，313/463达标。相比此前同上限最佳Goodput提高4.93%，最终目标仍未达到。\n\n'
        '真实运行代码保存在`../snapshot/root/SpecSLO/budget-calibration-20261015/`中的'
        '`tree-adaptive-own-history-full-v460/`、`tree-adaptive-retained-CANN-path-full-v461/`和'
        '`tree-adaptive-qualified-wait-margin-full-v462/`及其原依赖目录。'
        '新增独立检查和局部成本代码在`tree-adaptive-own-geometry-node-math-v459/`。\n\n'
        '源码逐字节保留。`SOURCE_MANIFEST.json`中的omitted_evidence明确列出未上传的历史数据与源码压缩档；'
        '仓库克隆不是原目录全量备份，不能直接继承旧NN许可或绕过原guard。恢复时按原条件重新生成fixture和证据并封存。'
        '模型与数据集、原始张量、图缓存和完整环境均未上传。证据摘要和原始收据的无损压缩副本位于evidence目录。\n'
    )
    manifest['files'] = sorted(saved.values(), key=lambda row: row['saved'])
    manifest['saved_UTC'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest['current_qualified_global64'] = validation
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    root_validation_path = ROOT / 'VALIDATION.json'
    root_validation = json.loads(root_validation_path.read_text())
    root_validation['ROOT462_qualified_full_result'] = validation
    root_validation['performance_goal_achieved'] = False
    root_validation_path.write_text(json.dumps(root_validation, ensure_ascii=False, indent=2) + '\n')
    readme = ROOT / 'README_zh.md'
    text = readme.read_text()
    old = '当前按用户要求将全局活动请求上限固定为64；v405完整463及独立own-AR已通过，Goodput439.1622；v409双物理批次完整候选已于2026-10-21 01:52 UTC启动。'
    new = '全局活动请求上限固定为64；当前v462完整463与三路独立own-AR通过，Raw681.6589、Goodput460.8191，313/463达标，实际峰值48。比此前同上限最佳v405提高4.93%；尚未超过固定γ4或最终1.5×目标。'
    assert old in text
    text = text.replace(old, new, 1)
    text = text.replace('| v398，全局64 |', '| v462，全局64 | 自适应M80/M120；原227+681、新693、75+6、完整463及三路own-AR通过 | 当前同上限最佳Goodput460.8191；实际峰值48，67.60%达标，仍需继续优化。 |\n| v398，全局64 |', 1)
    text = text.replace('v409双物理批次完整候选', '历史v409双物理批次候选')
    text += '\n当前已验证导出：`adaptive-M80-M120-full-v462/`。后续重点是按incumbent优先级、SLO余量和已到达prompt成本控制refill；未验证原型不标为性能提升。\n'
    readme.write_text(text)
    for relative, row in evidence.items():
        assert sha((DEST / relative).read_bytes()) == row['sha256']
    print(json.dumps(dict(status='PASS_EXACT462_CHECKPOINT_EXPORT', snapshot_files=len(saved),
                          copied_operations=len(copied), omitted_evidence=len(omitted), evidence_files=len(evidence),
                          Goodput=validation['Goodput'], goal_achieved=False)))


if __name__ == '__main__':
    main()
