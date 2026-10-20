"""Copy small terminal evidence and exact source bytes into the authorized branch."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
REPO = Path('/root/SpecSLO/checkpoints/github-SpecSLO-Tree')
ROOT = REPO / 'research/specslo-tree'


def copy_verified(source, destination, manifest):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
    manifest[destination.name if destination.parent.name == 'active64-segment-pack-v398'
             else 'tools/' + destination.name] = dict(source=str(source), sha256=digest)


def main():
    result = json.loads((D / 'ROOT398_FINAL_FULL_RESULT.json').read_text())
    assert result['whole463_own_AR_qualified'] is True
    dest = ROOT / 'active64-segment-pack-v398'
    manifest_path = dest / 'EXPORT_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    receipt = (B / 'tree-causal80-small-cohort-role-aware-audit-full-v209'
               / 'receipt-active64-whole-ticket-S24-segments-root-v398-full463-ownAR-v1.json')
    audit = D / 'ROOT398_TERMINAL_INDEPENDENT_FULL_AUDIT.json'
    compressed = []
    for source, name in [(audit, 'ROOT398_TERMINAL_INDEPENDENT_FULL_AUDIT.json.gz'),
                         (receipt, 'ROOT398_TERMINAL_SUPERVISOR_RECEIPT.json.gz')]:
        zipped = D / name
        zipped.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
        assert gzip.decompress(zipped.read_bytes()) == source.read_bytes()
        copy_verified(zipped, dest / 'tools' / name, manifest)
        compressed.append(dict(source=str(source), original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                               export='tools/' + name, compression='gzip',
                               compressed_sha256=hashlib.sha256(zipped.read_bytes()).hexdigest()))
    for name in [
            'ROOT398_FINAL_FULL_RESULT.json', 'ROOT398_OBSERVED_BUDGET_PRODUCTIVITY.json',
            'audit_ROOT398_budget_productivity_HOST.py', 'export_ROOT398_terminal_checkpoint_HOST.py',
            'ROOT398_ACTUAL_EAGER_COUNTS_OWN_AR_PENDING.json',
            'ROOT398_FORMAL_PHYSICAL_UTILIZATION_OWN_AR_PENDING.json']:
        copy_verified(D / name, dest / 'tools' / name, manifest)
    provenance = D / 'ROOT398_TERMINAL_EXPORT_PROVENANCE.json'
    provenance.write_text(json.dumps(dict(compressed_evidence=compressed,
        raw_tensors_or_graph_cache_exported=False, neural_source_bytes_unchanged=True,
        model_or_dataset_copied=False, previous_pending_count_filenames_retained=True,
        pending_filenames_describe_extraction_time_not_final_qualification=True), indent=2) + '\n')
    copy_verified(provenance, dest / 'tools' / provenance.name, manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    validation_path = dest / 'VALIDATION.json'
    validation = json.loads(validation_path.read_text())
    validation.update(status='FULL398_GLOBAL64_COMPLETE_INDEPENDENTLY_AUDITED_REGRESSION_RETAINED',
                      NPU_full_run_completed=True, Raw=result['Raw'], Goodput=result['Goodput'],
                      attained=result['attained'], requests=463, actual_active_peak=64,
                      independent_own_AR_targets=3, exact_output_tokens_per_target=118528,
                      additional_uniform_KV_math_per_target=681, original_math_per_target=227,
                      performance_goal_met=False, adopted=False, matched_repeats=0,
                      historical609_result_active_limit=48,
                      next_scope='finite B4 diagnostic; then prioritize frugal S32 NN/cost; preserve global64')
    validation_path.write_text(json.dumps(validation, indent=2) + '\n')
    dest400 = ROOT / 'frugal399-guard-repair-v400'
    path400 = dest400 / 'EXPORT_MANIFEST.json'
    manifest400 = json.loads(path400.read_text())
    copy_verified(D / 'audit_ROOT400_finite399_terminal_HOST.py',
                  dest400 / 'tools/audit_ROOT400_finite399_terminal_HOST.py', manifest400)
    path400.write_text(json.dumps(manifest400, indent=2) + '\n')
    report = Path('/root/SpecSLO/tpot-report-20261015/DATASET_TPOT_AND_OPTIMIZATION_REPORT_zh.md')
    report_dest = ROOT / 'snapshot/root/SpecSLO/tpot-report-20261015' / report.name
    shutil.copyfile(report, report_dest)
    snapshot_path = ROOT / 'SOURCE_MANIFEST.json'
    snapshot = json.loads(snapshot_path.read_text())
    entries = [item for item in snapshot['files']
               if item['saved'] == str(report_dest.relative_to(REPO))]
    assert len(entries) == 1
    entries[0].update(bytes=report_dest.stat().st_size,
                      sha256=hashlib.sha256(report_dest.read_bytes()).hexdigest())
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    for folder, saved in [(dest, manifest), (dest400, manifest400)]:
        for relative, item in saved.items():
            assert hashlib.sha256((folder / relative).read_bytes()).hexdigest() == item['sha256']
    print(json.dumps(dict(status='PASS_TERMINAL_EXPORT_BYTE_HASHES', files398=len(manifest),
                          files400=len(manifest400), compressed_bytes=sum((D / Path(r['export']).name).stat().st_size
                                                                         for r in compressed),
                          Raw=result['Raw'], Goodput=result['Goodput'])))


if __name__ == '__main__':
    main()
