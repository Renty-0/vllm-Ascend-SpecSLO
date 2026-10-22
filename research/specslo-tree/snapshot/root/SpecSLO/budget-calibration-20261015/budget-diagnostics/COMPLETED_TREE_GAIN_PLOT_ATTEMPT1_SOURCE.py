#!/usr/bin/env python3
"""Plot saved full463 adaptive gain observations after an exact passed whole audit.

Uses one of four identical cooperating-rank snapshots, never their sum. Missing
observations remain blank. Existing Pillow/Matplotlib only; no package install,
Torch, tensor input, model import, workload, or device query.
"""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import types

D = Path(__file__).resolve().parent
PINS = {
    'plot_v140_actual_gain_host.py': '9a99065d23a191e5cef2ad2e4333d8ed6d62bbfef1f0c6dc566964e1ed93e925',
    'analyze_completed_tree_full463_cost.py': 'd3266647799fc0cf00c49fe98e195ecb85dc446e51ab66c643a5b1d5c3ca0b74',
}
FORBIDDEN = {'torch', 'torch_npu', 'vllm', 'vllm_ascend', 'transformers'}


def need(ok, message):
    if not ok:
        raise ValueError('audited gain plot: ' + message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(part)
    return digest.hexdigest()


def desc(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


class NoModelImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise ImportError('HOST gain plotting forbids ' + fullname)
        return None


def load_pinned_helpers():
    for name, wanted in PINS.items():
        need(sha(D / name) == wanted, 'pinned helper source ' + name)
    spec = importlib.util.spec_from_file_location('_completed_tree_gain_audit_inputs', D / 'analyze_completed_tree_full463_cost.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    # Reuse the unchanged renderer with only its two display titles extended.
    # The old source and old figures are never written.
    original = (D / 'plot_v140_actual_gain_host.py').read_text()
    old = 'f"{cls} | {provenance[\'phase\']} | generation {provenance[\'source_generation\']}"'
    new = 'f"{cls} | {provenance[\'phase\']} | generation {provenance[\'source_generation\']} | adaptive (not causal)"'
    need(original.count(old) == 2 and new not in original, 'two exact renderer title anchors')
    patched = original.replace(old, new)
    need(patched.replace(new, old) == original, 'renderer title-only inverse')
    renderer = types.ModuleType('_audited_tree_gain_renderer')
    renderer.__file__ = str(D / 'plot_v140_actual_gain_host.py')
    exec(compile(patched, renderer.__file__, 'exec'), renderer.__dict__)
    return helper, renderer, {
        'original': desc(D / 'plot_v140_actual_gain_host.py'),
        'in_memory_change': 'Exactly two display-title suffixes: adaptive (not causal)',
        'exact_inverse_verified': True,
        'patched_renderer_source_sha256': hashlib.sha256(patched.encode()).hexdigest(),
        'original_renderer_or_figures_modified': False,
    }


def crosscheck_csv(path, wanted, rows):
    need(path is not None and wanted is not None, 'reference CSV path and SHA are a pair')
    need(sha(path) == wanted, 'reference CSV exact SHA')
    with path.open(newline='') as stream:
        previous = list(csv.DictReader(stream))
    keys = lambda row: (row['slo_class'], int(row['B']))
    index = {keys(row): row for row in previous}
    need(len(index) == len(previous) == len(rows) == 15, 'reference CSV unique complete class/B domain')
    integer_fields = ('scope_count', 'source_generation', 'observed_rounds', 'observed_delivered', 'policy_prior_weight')
    numeric_fields = ('observed_g', 'policy_prior_g', 'policy_intrinsic_posterior_g')
    compared = 0
    for row in rows:
        old = index[keys(row)]
        need(old['phase'] == row['phase'], 'reference phase')
        for field in integer_fields + numeric_fields:
            converter = int if field in integer_fields else float
            value = converter(old[field]) if old[field] != '' else None
            need(value == row[field], 'reference CSV exact saved count/value ' + str(keys(row)) + '/' + field)
            compared += 1
    return {'reference': desc(path), 'matched_rows': len(rows), 'matched_numeric_fields': compared,
            'rank_column_not_compared': 'Four ranks cooperate; selected rank identity is not an independent trial'}


def write_csv(path, rows, fields):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    return desc(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch-directory', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--expected-receipt-sha256', required=True)
    parser.add_argument('--actual-audit', type=Path, required=True)
    parser.add_argument('--expected-audit-sha256', required=True)
    parser.add_argument('--rank', type=int, choices=range(4), default=1)
    parser.add_argument('--include-policy-estimates', action='store_true')
    parser.add_argument('--reference-csv', type=Path)
    parser.add_argument('--expected-reference-csv-sha256')
    parser.add_argument('--output-name', required=True, help='New simple directory name below budget-diagnostics')
    args = parser.parse_args()
    need(all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', value) for value in (args.tag, args.output_name)), 'private tag/output name')
    need(os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') == '0'
         and all(os.environ.get(name) == '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')), 'HOST backend/thread caps')
    need(not any(name.split('.')[0] in FORBIDDEN for name in sys.modules), 'no preloaded model/backend imports')
    sys.meta_path.insert(0, NoModelImports())
    output = D / args.output_name
    need(not output.exists(), 'refusing to overwrite an existing output directory')
    need((args.reference_csv is None) == (args.expected_reference_csv_sha256 is None), 'reference CSV path/SHA pair')
    helper, renderer, renderer_change = load_pinned_helpers()
    # This helper's loader is stdlib SHA/JSON validation only. Its Native-source
    # AST/clock analysis main is not called, so existing system Pillow is usable.
    epoch, receipt, audit, freeze, reports, traces, envelopes, inputs = helper.load_inputs(args)
    generations = [envelope['source_generation'] for envelope in envelopes]
    need(len(set(generations)) == 1, 'four actual generation identities')
    snapshots = [renderer.normalize_gain(envelope['actual_gain_state']['after'], generations[rank])
                 for rank, envelope in enumerate(envelopes)]
    need(all(snapshot == snapshots[args.rank] for snapshot in snapshots), 'four identical gain/prior snapshots')
    provenance = {
        'directory': str(epoch), 'tag': args.tag, 'phase': 'full463', 'scope_count': 463,
        'source_generation': generations[args.rank], 'rank_used': args.rank,
        'receipt': inputs['receipt'], 'passed_whole_external_audit': inputs['actual_audit'],
        'phase_inputs': inputs, 'cross_rank_gain_state_equal': True,
        'four_ranks_are_independent_trials': False, 'four_rank_snapshots_summed': False,
    }
    rows = renderer.table_rows(provenance, snapshots[args.rank])
    expected = snapshots[args.rank][0]
    actual = {(row['slo_class'], row['B']): (row['observed_rounds'], row['observed_delivered'])
              for row in rows if row['observed_rounds'] is not None}
    need(actual == expected, 'CSV contains exactly one saved observed counter snapshot')
    crosscheck = crosscheck_csv(args.reference_csv, args.expected_reference_csv_sha256, rows) if args.reference_csv else None
    need(importlib.util.find_spec('matplotlib') is not None or importlib.util.find_spec('PIL') is not None,
         'use an existing Matplotlib/Pillow interpreter; no dependency installation')
    # Validate all input and reference data before creating any new output.
    output.mkdir(exist_ok=False)
    files = [write_csv(output / 'actual-observed-g.csv', rows, list(rows[0]))]
    observed_fields = ['phase', 'scope_count', 'source_generation', 'rank_used', 'slo_class', 'B',
                       'observed_rounds', 'observed_delivered', 'observed_g']
    for cls in renderer.CLASSES:
        files.append(write_csv(output / (cls + '-observed-g.csv'),
                               [row for row in rows if row['slo_class'] == cls], observed_fields))
    files.append(write_csv(output / 'policy-prior-and-posterior.csv', rows,
                           observed_fields[:6] + ['policy_prior_g', 'policy_prior_weight', 'policy_intrinsic_posterior_g']))
    images, backend = renderer.plots(rows, provenance, output, args.include_policy_estimates)
    files.extend(images)
    need(not any(name.split('.')[0] in FORBIDDEN for name in sys.modules), 'no model imports after plotting')
    manifest = {
        'schema': 'HOST.completed-tree-actual-gain-plots.v1', 'status': 'passed',
        'UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'actual_HOST_argv': sys.orig_argv,
        'python_version': sys.version, 'script': desc(__file__),
        'input_validator': desc(D / 'analyze_completed_tree_full463_cost.py'),
        'renderer': renderer_change, 'plotting_backend': backend, 'provenance': provenance,
        'source_saved_gain_after': envelopes[args.rank]['actual_gain_state']['after'],
        'source_saved_observed_counters_reproduced_exactly': True, 'crosscheck': crosscheck,
        'outputs': files, 'policy_estimates_drawn_with_separate_labels': args.include_policy_estimates,
        'unobserved_class_B': [{'slo_class': row['slo_class'], 'B': row['B']}
                              for row in rows if row['observed_rounds'] is None],
        'definitions': {
            'observed_g': 'Saved actual delivered / rounds; no prior in this value',
            'prior': 'Actual saved calibrated policy.base[class][B], not current-generation measurements',
            'posterior': '(12*prior + delivered)/(12+rounds), only at actually observed B; not EMA',
            'missing': 'Unobserved rounds/delivered/g/posterior are blank, including B1/B2 if absent',
            'scope': 'Adaptive selected, correlated observations; neither controlled B comparison nor causal gain',
            'CI': 'None; these cooperating-rank adaptive observations are not independent trials',
            'request_local_clamping': 'Remaining-token clamp is not reconstructed by this plotting tool',
        },
        'formal_Goodput_or_new_numerical_qualification_inferred': False,
        'NPU_Torch_tensor_model_or_device_query_executed': False,
        'frozen_sources_bridge_or_old_plots_modified': False,
    }
    path = output / 'provenance.json'
    with path.open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output_directory': str(output), 'observed_class_B': len(expected),
                      'observed_rounds_one_snapshot': sum(value[0] for value in expected.values()),
                      'CSV_count': 5, 'PNG_count': 3, 'SVG_count': 3, 'crosscheck': crosscheck,
                      'provenance': desc(path), 'V143_actual_read': epoch.name.endswith('v143')}))


if __name__ == '__main__':
    main()
