"""Save exact research sources without datasets, caches, or runtime mutation."""
import datetime
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

BASE = Path('/root/SpecSLO/budget-calibration-20261015')
REPO = Path('/root/SpecSLO/checkpoints/github-SpecSLO-Tree')
DEST = REPO / 'research/specslo-tree'
SOURCES = (
    ('tree-static-q5-M120-SLO-first-full463-own-AR-v257', 257),
    ('tree-dynamic-root-alternate-eager-compat-full-v287', 287),
    ('tree-all-layer-prefix-dual-cache-target-cost-v291', 291),
    ('tree-safe-grid-span-dual-cache-probe-v295', 295),
    ('tree-concrete-device-span-dual-cache-target-cost-v296', 296),
    ('tree-capacity-request-risk-span-full-v299', 299),
    ('tree-single-prepare-target-math-cost-v311', 311),
    ('tree-measured-cost-risk-single-prepare-full-v316', 316),
    ('tree-M165-N32-independent-target-math-cost-v317', 317),
    ('tree-M165-raw-reference-api-v318', 318),
    ('tree-M165-NZ166-target-math-cost-v320', 320),
    ('tree-M165-first-divergence-v321', 321),
    ('tree-M165-first-divergence-v322', 322),
    ('tree-M165-common-mask11-target-math-cost-v323', 323),
    ('tree-M160-S32-N32-target-math-cost-v326', 326),
    ('tree-M160-C128-capacity-binding-v329', 329),
    ('tree-M160-actual165-MC2-bridge-host-v330', 330),
    ('tree-FIA160-projection165-MC2-target-math-cost-v331', 331),
    ('tree-FIA160-projection165-MC2-audit-adapter-v332', 332),
    ('tree-M160-C128-F64-global192-full-v333', 333),
    ('tree-FIA160-projection165-chained-flush-host-v334', 334),
    ('tree-FIA-Q4-group-and-query-isolation-v335', 335),
    ('tree-Q4-linear-target-math-cost-v336', 336),
    ('tree-Q4-original-output-root-adapter-v337', 337),
    ('tree-Q4-actual-host-builder-adapter-v338', 338),
    ('tree-Q4-closure-aware-host-builder-adapter-v339', 339),
    ('tree-FIA-Q5-causal-uniform-KV-screen-v340', 340),
    ('tree-Q5-bucket-causal-inner1-target-math-cost-v341', 341),
    ('tree-Q5-uniform-KV-bucket-only-target-math-cost-v342', 342),
    ('tree-Q5-uniform-KV-bucket-branch-math-cost-v343', 343),
    ('tree-Q5-uniform-KV-bucket-branch-math-cost-v344', 344),
    ('tree-coalesced-live-refill-host-v345', 345),
    ('tree-coalesced-live-refill-M120-B8-full-v346', 346),
    ('tree-small-refill-singleton-graph-capture-v347', 347),
    ('tree-small-refill-singleton-module-binding-v349', 349),
    ('tree-small-refill-full-next-subset-domain-v350', 350),
    ('tree-small-refill-incumbent-prefix-protection-v351', 351),
    ('tree-small-refill-original-launch-pin-v352', 352),
    ('tree-coalesced-live-refill-M120-singleton-capture-full-v348', 348),
    ('tree-admission-closed-control-full-v355', 355),
    ('tree-priority-protected-refill-full-v356', 356),
    ('tree-Q5-bucket-completion-and-paired-mode-cost-v361', 361),
    ('tree-FIA-multi-owner-page-offset-invariance-v363', 363),
    ('tree-primary4-top2-chain-device-v365', 365),
    ('tree-primary4-top2-chain-owned-guard-v366', 366),
    ('tree-primary4-top2-chain-runtime-init-v367', 367),
    ('tree-primary4-top2-chain-compiler-cache-v368', 368),
    ('tree-primary4-tail3-top2-chain-device-v370', 370),
    ('tree-primary4-tail3-eager-allN-device-v371', 371),
    ('tree-primary4-tail3-best257-full-v372', 372),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selected = set()
    omitted = {}
    freezes = []
    for directory, version in SOURCES:
        path = BASE / directory / f'FREEZE{version}.json'
        freeze = json.loads(path.read_text())
        freezes.append(dict(source=str(path), sha256=digest(path)))
        selected.add(path)
        for filename, expected in freeze['files'].items():
            source = Path(filename)
            if any(part in {'HOST-attempts', 'HOST-loop-attempts'} for part in source.parts):
                omitted[filename] = dict(sha256=expected, reason='Historical HOST attempt artifact')
            elif source.suffix in {'.py', '.sh', '.pre_descriptor_update'}:
                if digest(source) != expected:
                    raise RuntimeError(f'Frozen source changed: {source}')
                selected.add(source)
            else:
                omitted[filename] = dict(sha256=expected, reason='Historical evidence or fixture; not source')
    for directory in (
        'tree-prefix-precopy-dual-cachewrite-host-v288',
        'tree-spare-capacity-budget-host-v290',
        'tree-contiguous-prefix-span-copy-host-v292',
        'tree-request-conditional-risk-gain-host-v294',
        'tree-capacity-request-risk-span-full-v299',
        'tree-single-validated-prefix-descriptors-host-v304',
        'tree-single-validated-prefix-prepare-host-v305',
        'tree-capacity-risk-single-prepare-full-v307',
        'tree-range-packed-branch-layout-host-v308',
        'tree-measured-branch-cost-eligibility-host-v309',
        'tree-single-prepare-target-math-cost-v311',
        'tree-uniform-q5-M165-S33-C128-host-v285',
        'tree-M165-single-prepare-host-v313',
        'tree-C128-F64-global192-budget-host-v314',
        'tree-current-prepare-cost-eligibility-host-v312',
        'tree-measured-cost-risk-single-prepare-full-v316',
        'tree-M165-N32-independent-target-math-cost-v317',
        'tree-M165-raw-reference-api-v318',
        'tree-C128-F64-global192-integration-host-v319',
        'tree-M165-NZ166-target-math-cost-v320',
        'tree-M165-first-divergence-v321',
        'tree-M165-first-divergence-v322',
        'tree-M165-common-mask11-target-math-cost-v323',
        'tree-FIA-group-root-isolation-v324',
        'tree-M165-common-mask11-reference-fix-v325',
        'tree-M160-S32-N32-target-math-cost-v326',
        'tree-M160-MC2-memory-qualification-host-v327',
        'tree-C128-F64-global192-groups32-host-v328',
        'tree-M160-C128-capacity-binding-v329',
        'tree-M160-actual165-MC2-bridge-host-v330',
        'tree-FIA160-projection165-MC2-target-math-cost-v331',
        'tree-FIA160-projection165-MC2-audit-adapter-v332',
        'tree-M160-C128-F64-global192-full-v333',
        'tree-FIA160-projection165-chained-flush-host-v334',
        'tree-FIA-Q4-group-and-query-isolation-v335',
        'tree-Q4-linear-target-math-cost-v336',
        'tree-Q4-original-output-root-adapter-v337',
        'tree-Q4-actual-host-builder-adapter-v338',
        'tree-Q4-closure-aware-host-builder-adapter-v339',
        'tree-FIA-Q5-causal-uniform-KV-screen-v340',
        'tree-Q5-bucket-causal-inner1-target-math-cost-v341',
        'tree-Q5-uniform-KV-bucket-only-target-math-cost-v342',
        'tree-Q5-uniform-KV-bucket-branch-math-cost-v343',
        'tree-Q5-uniform-KV-bucket-branch-math-cost-v344',
        'tree-coalesced-live-refill-host-v345',
        'tree-coalesced-live-refill-M120-B8-full-v346',
        'tree-small-refill-singleton-graph-capture-v347',
        'tree-small-refill-singleton-module-binding-v349',
        'tree-small-refill-full-next-subset-domain-v350',
        'tree-small-refill-incumbent-prefix-protection-v351',
        'tree-small-refill-original-launch-pin-v352',
        'tree-coalesced-live-refill-M120-singleton-capture-full-v348',
        'tree-priority-protected-live-refill-host-v354',
        'tree-admission-closed-control-full-v355',
        'tree-priority-protected-refill-full-v356',
        'tree-depth3-branch6-protocol-host-v357',
        'tree-Q4S32-branch-projection-host-v358',
        'tree-Q4S32-prefix-alias-split-host-v359',
        'tree-Q4S32-depth3-target-adapter-host-v360',
        'tree-Q5-bucket-completion-and-paired-mode-cost-v361',
        'tree-Q5-multi-owner-page-table-packing-host-v362',
        'tree-FIA-multi-owner-page-offset-invariance-v363',
        'tree-primary4-top2-chain-host-v364',
        'tree-primary4-top2-chain-device-v365',
        'tree-primary4-top2-chain-owned-guard-v366',
        'tree-primary4-top2-chain-runtime-init-v367',
        'tree-primary4-top2-chain-compiler-cache-v368',
        'tree-primary4-tail3-top2-chain-host-v369',
        'tree-primary4-tail3-top2-chain-device-v370',
        'tree-primary4-tail3-eager-allN-device-v371',
        'tree-primary4-tail3-best257-full-v372',
    ):
        selected.update(p for p in (BASE / directory).iterdir()
                        if p.suffix in {'.py', '.sh', '.json'} and not p.name.startswith('FIXTURE'))
    for filename in (
        'audit_ROOT287_declared_B1to8_HOST.py',
        'audit_ROOT295_operator_evidence_HOST.py',
        'calibrate_ROOT287_same_prefix_gains_HOST.py',
        'plot_ROOT287_same_prefix_gains_HOST.py',
        'read_common_publication_metrics.py',
        'common_publication_metrics.py',
        'final_candidate_eligibility.py',
        'prepare_final_comparison_plan.py',
        'final_common_comparison.py',
        'final_owned_processes.py',
        'final_baseline_cell.py',
        'ROOT287_INDEPENDENT_FULL_DOMAIN8_AUDIT.json',
        'ROOT295_INDEPENDENT_OPERATOR_AUDIT.json',
        'ROOT287_B8_COHORT_SAME_PREFIX_GAINS_AR_QUALIFIED.json',
        'ROOT_FINAL_THREE_BLOCK_COMPARISON_PREDECLARATION_20261018.json',
        'ROOT293_HARDWARE_GRID_LIMIT_FAILURE.json',
        'ROOT291_INITIALIZATION_DEVICE_FAILURE.json',
        'audit_ROOT296_target_math_and_cost_HOST.py',
        'build_ROOT299_capacity_risk_span_full_HOST.py',
        'freeze_ROOT299_capacity_risk_span_full_HOST.py',
        'save_ROOT_tree_github_snapshot.py',
        'ROOT296_INDEPENDENT_TARGET_MATH_AND_COST_AUDIT.json',
        'ROOT296_VS280_HISTORICAL_TARGET_COST.json',
        'build_ROOT305_single_validated_prepare_HOST.py',
        'build_ROOT307_fast_prepare_full_HOST.py',
        'freeze_ROOT307_fast_prepare_full_HOST.py',
        'monitor_ROOT299_readonly_HOST.py',
        'audit_ROOT299_terminal_full_HOST.py',
        'ROOT_FINAL_ELIGIBILITY_B8_DOMAIN_REGRESSION.json',
        'ROOT299_TERMINAL_GEOMETRY_HOST_REGRESSION.json',
        'ROOT_NEXT_PERFORMANCE_LEVERS_20261018.json',
        'profile_ROOT305_prepare_HOST.py',
        'ROOT305_CPU_PREPARE_PROFILE.json',
        'ROOT257_287_ACTUAL_DELIVERY_MARGIN_AND_PHASES.json',
        'ROOT299_PRELIMINARY_FULL_COMMON_METRICS.json',
        'ROOT299_FINAL_FULL_RESULT.json',
        'ROOT299_TERMINAL_INDEPENDENT_FULL_AUDIT.json',
        'build_ROOT311_single_prepare_target_cost_HOST.py',
        'freeze_ROOT311_single_prepare_target_cost_HOST.py',
        'audit_ROOT311_target_math_and_cost_HOST.py',
        'build_ROOT313_M165_prepare_HOST.py',
        'ROOT311_INDEPENDENT_TARGET_MATH_AND_COST_AUDIT.json',
        'ROOT311_VS296_TARGET_PREPARE_COST_COMPARISON.json',
        'ROOT_M165_N32_NEXT_DEVICE_PLAN_20261019.json',
        'build_ROOT316_measured_cost_full_HOST.py',
        'freeze_ROOT316_measured_cost_full_HOST.py',
        'audit_ROOT316_terminal_full_HOST.py',
        'ROOT316_GENERATED_SOURCE_NUMERIC_INTEGRITY.json',
        'monitor_ROOT316_readonly_HOST.py',
        'ROOT316_FINAL_FULL_RESULT.json',
        'ROOT316_PRELIMINARY_FULL_COMMON_METRICS.json',
        'ROOT316_TERMINAL_INDEPENDENT_FULL_AUDIT.json',
        'freeze_ROOT317_M165_target_cost_HOST.py',
        'freeze_ROOT317_ABI311_launch_HOST.py',
        'freeze_ROOT317_ABI311_paths_HOST.py',
        'summarize_ROOT_official021_stability_HOST.py',
        'ROOT316_ACTUAL_ALL_GAMMA4_CALL_COUNTS.json',
        'ROOT_TREE_RAW_FOLLOWUP_WITH_ORIGINAL_PLAN_20261019.json',
        'ROOT318_UNUSED_COMPILER_CACHE_REGENERATION_20261019.json',
        'ROOT318_320_FIRST_BRANCH_FAILURE_COMPARISON.json',
        'ROOT322_FIRST_DIVERGENCE_INDEPENDENT_READ.json',
        'build_ROOT323_common_mask_HOST.py',
        'ROOT316_EAGER_B4_EXCLUSIONS_RAW_FOLLOWUP_20261019.json',
        'build_ROOT326_M160_HOST.py',
        'ROOT326_PARTIAL_MODEL_CAPACITY_FAILURE_AUDIT.json',
        'ROOT_M160_MC2_IN_MEMORY_QUALIFICATION_PLAN_20261019.json',
        'ROOT_M160_TO_QUALIFIED_M165_MC2_BRIDGE_PLAN_20261019.json',
        'ROOT_FUTURE_RAW_STATIC_KV_BUCKET_PLAN_20261019.json',
        'audit_ROOT_M160_math_cost_HOST.py',
        'check_ROOT331_runtime_import_HOST.py',
        'startup_ROOT331_runtime_import_HOST.sh',
        'build_ROOT333_M160_full_HOST.py',
        'integrate_ROOT333_geometry_HOST.py',
        'freeze_ROOT333_current_cost_full_HOST.py',
        'startup_ROOT333_actual_install_HOST.sh',
        'cleanup_ROOT331_reconstructable_private_HOST.py',
        'audit_ROOT331_rejected_observations_HOST.py',
        'ROOT329_INDEPENDENT_TARGET_MATH_AND_COST_AUDIT.json',
        'ROOT331_REJECTED_RUN_RETAINED_COMPONENT_OBSERVATIONS.json',
        'ROOT331_334_ABI_SOURCE_REVIEW_20261019.json',
        'ROOT329_331_REDUNDANT_CHECKPOINT_AND_OLD_CACHE_REGENERATION_20261019.json',
        'ROOT316_NONTERMINAL_LOW_B_RAW_CHECK_20261019.json',
        'audit_ROOT333_terminal_full_HOST.py',
        'ROOT333_TERMINAL_READER_SOURCE_PREPARED.json',
        'HOST_ROOT333_TERMINAL_READER_RESULT.json',
        'monitor_ROOT333_readonly_HOST.py',
        'analyze_ROOT_tree_terminal_gain_HOST.py',
        'ROOT316_TERMINAL_GAIN_DESCRIPTIVE_20261019.json',
        'ROOT_FUTURE_Q4_NO_BONUS_RAW_HYPOTHESIS_20261019.json',
        'ROOT_Q4_FUTURE_RAW_HISTORICAL_BONUS_COST_SCREEN_20261019.json',
        'HOST_ROOT333_TERMINAL_READER_LEGACY_PROJECTION_RESULT.json',
        'ROOT333_posthoc_projection_adapter_HOST.py',
        'audit_ROOT333_terminal_corrected_projection_HOST.py',
        'HOST_ROOT333_ACTUAL_LOGICAL_AND_DEVICE_PROJECTION_REGRESSION.json',
        'ROOT333_TERMINAL_ORIGINAL_READER_AUDIT.json',
        'ROOT333_TERMINAL_INDEPENDENT_CORRECTED_PROJECTION_AUDIT.json',
        'ROOT333_CURRENT_COST_OBSERVATIONS_FULL_PENDING_20261019.json',
        'ROOT333_PRELIMINARY_FULL_COMMON_METRICS.json',
        'ROOT333_PRELIMINARY_FULL_RESULT_AND_RAW_FINDINGS.json',
        'ROOT333_FINAL_FULL_RESULT.json',
        'ROOT_FUTURE_Q4_DEPTH3_EXISTING_EAGER_RAW_ALTERNATIVE_20261019.json',
        'ROOT_FUTURE_RISK_CPU_PROFILE_RAW_PLAN_20261019.json',
        'ROOT336_OUTPUT_ROOT_STARTUP_FAILURE.json',
        'ROOT337_ACTUAL_FRONTEND_BYPASS_FAILURE.json',
        'ROOT338_ACTUAL_CHECKED_FRONTEND_WRAPPER_FAILURE.json',
        'audit_ROOT339_retained_component_HOST.py',
        'ROOT339_INDEPENDENT_LINEAR_MATH_AND_COST_AUDIT.json',
        'ROOT257_ACTUAL_TARGET_CONTEXT_EXTENT_RAW_SCREEN_20261019.json',
        'ROOT340_SYNTHETIC_UNIFORM_KV_CAUSAL_SCREEN.json',
        'ROOT341_CAUSAL_CROSS_FULL_HIDDEN_FAILURE.json',
        'ROOT257_EMPTY_ACTIVE_ADMISSION_RAW_FOLLOWUP_20261019.json',
        'ROOT342_INDEPENDENT_FULL_BUCKET_MODEL_COST_AUDIT.json',
        'ROOT343_NEGATIVE_CONTROL_OBSERVER_FAILURE.json',
        'ROOT346_ARCHIVAL_FORMAT_CHECK_EXCEPTION.json',
        'ROOT344_PREDECLARED_DEADLINE_PARTIAL_MATH.json',
        'audit_ROOT344_partial_retained_math_HOST.py',
        'ROOT344_INDEPENDENT_PARTIAL_RETAINED_MATH_AUDIT.json',
        'audit_ROOT346_terminal_full_HOST.py',
        'check_ROOT346_terminal_source_HOST.py',
        'HOST_ROOT346_TERMINAL_SOURCE_RESULT.json',
        'read_ROOT346_common_result_HOST.py',
        'HOST_ROOT346_COMMON_RESULT_READER.json',
        'ROOT346_SINGLETON_CAPTURE_PROTOCOL_FAILURE.json',
        'ROOT347_SUBSET_MODULE_BINDING_FAILURE.json',
        'ROOT349_OWNED_ABORT_FOR_MISSING_NEXT_DRAFT_DOMAIN.json',
        'ROOT346_HOST_INITIAL_PRIOR_COST_GATE_SCREEN.json',
        'ROOT350_EMPTY_NONSELECTED_PREFIX_GUARD_FAILURE.json',
        'ROOT351_LAUNCH_PIN_SUBSTRING_RENAME_FAILURE.json',
        'ROOT352_MISSING_ORIGINAL_TAIL_OBSERVER_FAILURE.json',
        'audit_ROOT348_terminal_full_HOST.py',
        'check_ROOT348_terminal_source_HOST.py',
        'HOST_ROOT348_TERMINAL_SOURCE_RESULT.json',
        'read_ROOT348_common_result_HOST.py',
        'profile_ROOT_risk294_owner_HOST.py',
        'ROOT_RISK294_LOOKUP_ONLY_CPU_PROFILE_20261019.json',
        'ROOT_RISK294_LOOKUP_ONLY_CPU_PROFILE_WITH_COST_GATE_20261019.json',
        'monitor_ROOT348_readonly_HOST.py',
        'analyze_ROOT_live_refill_raw_HOST.py',
        'ROOT257_REFILL_RAW_READER_HOST_CONTROL.json',
        'ROOT348_RAW_ADMISSION_METRICS_PREDECLARATION_20261019.json',
        'ROOT_PREPARED_FINAL_REFERENCE_SOURCE_READINESS_20261019.json',
        'audit_ROOT348_small_refill_component_HOST.py',
        'ROOT348_INDEPENDENT_SMALL_REFILL_COMPONENT_AUDIT.json',
        'ROOT348_RETAINED_408_TARGET_COMPONENT_CHECKPOINT.json',
        'ROOT348_PRELIMINARY_FULL_COMMON_METRICS.json',
        'ROOT348_TERMINAL_INDEPENDENT_FULL_AUDIT.json',
        'ROOT348_FINAL_FULL_RESULT.json',
        'ROOT348_VS257_REFILL_RAW_DESCRIPTIVE.json',
        'ROOT348_LIVE_REFILL_INCUMBENT_EXPOSURES.json',
        'build_ROOT355_closed_control_HOST.py',
        'freeze_ROOT355_closed_control_HOST.py',
        'HOST_ROOT355_FROZEN_RUNTIME_CONSTRUCTION.json',
        'audit_ROOT355_terminal_full_HOST.py',
        'read_ROOT355_common_result_HOST.py',
        'check_ROOT355_terminal_source_HOST.py',
        'HOST_ROOT355_TERMINAL_SOURCE_RESULT.json',
        'ROOT355_CLOSED_ADMISSION_CONTROL_PREDECLARATION.json',
        'monitor_ROOT355_readonly_HOST.py',
        'build_ROOT356_protected_refill_HOST.py',
        'freeze_ROOT356_protected_refill_HOST.py',
        'HOST_ROOT356_FROZEN_RUNTIME_CONSTRUCTION.json',
        'audit_ROOT356_terminal_full_HOST.py',
        'read_ROOT356_common_result_HOST.py',
        'check_ROOT356_terminal_source_HOST.py',
        'HOST_ROOT356_TERMINAL_SOURCE_RESULT.json',
        'ROOT356_PROTECTED_REFILL_PREDECLARATION.json',
        'ROOT_Q4S32_LIVE_POOL_FOLLOWUP_SCREEN_20261019.json',
        'calibrate_ROOT357_same_prefix_short_tree_HOST.py',
        'ROOT357_CONDITIONAL_DEPTH3_GAIN_SCREEN.json',
        'ROOT355_PRELIMINARY_FULL_COMMON_METRICS.json',
        'ROOT355_FINAL_FULL_RESULT.json',
        'ROOT355_TERMINAL_INDEPENDENT_FULL_AUDIT.json',
        'monitor_ROOT356_readonly_HOST.py',
        'build_ROOT361_bucket_completion_guard_HOST.py',
        'freeze_ROOT361_bucket_completion_HOST.py',
        'check_ROOT361_actual_frozen_construction_HOST.py',
        'HOST_ROOT361_ACTUAL_FROZEN_CONSTRUCTION.json',
        'ROOT356_PRELIMINARY_FULL_COMMON_METRICS.json',
        'ROOT356_FINAL_FULL_RESULT.json',
        'ROOT356_TERMINAL_INDEPENDENT_FULL_AUDIT.json',
        'ROOT363_NEGATIVE_SYNTHETIC_RESULT.json',
        'ROOT365_367_LAUNCH_FAILURES_RETAINED.json',
        'ROOT368_INDEPENDENT_REAL_DRAFT_COMPONENT_RESULT.json',
        'ROOT370_INDEPENDENT_REAL_DRAFT_TAIL3_RESULT.json',
        'ROOT371_INDEPENDENT_REAL_DRAFT_EAGER_RESULT.json',
        'ROOT372_FULL_TAIL3_PLAN_20261019.json',
        'monitor_ROOT372_readonly_HOST.py',
        'ROOT355_CONCURRENT_HOST_VALIDATION_TIMING_CONTEXT.json',
    ):
        path = BASE / 'budget-diagnostics' / filename
        if path.exists():
            selected.add(path)
    official = Path('/root/SpecSLO/official-v021-baseline-20261019')
    selected.update(p for p in official.iterdir() if p.is_file()
                    and p.suffix in {'.py', '.sh', '.json', '.md'}
                    and not p.name.startswith('SOURCE_MANIFEST')
                    and not p.name.startswith(('HOST', 'SCOPED_BUILD_ATTEMPT')))
    for directory in ('harness', 'publication-witness'):
        selected.update(p for p in (official / directory).rglob('*') if p.is_file()
                        and '__pycache__' not in p.parts and 'actual' not in p.parts
                        and p.suffix in {'.py', '.json'}
                        and (directory == 'harness' or p.parent == official / directory))
    selected.add(BASE / 'budget-diagnostics' / 'ROOT_20261019_OLD_OFFICIAL_BASELINE_AND_TREE_RESUME_HANDOFF.json')
    selected.add(Path('/root/SpecSLO/tpot-report-20261015/DATASET_TPOT_AND_OPTIMIZATION_REPORT_zh.md'))
    plugin = Path('/root/data/vllm-ascend-hust')
    for directory in ('vllm_ascend/spec_decode', 'vllm_ascend/ops/triton', 'csrc/mc2'):
        selected.update(p for p in (plugin / directory).rglob('*')
                        if p.is_file() and not p.is_symlink()
                        and '__pycache__' not in p.parts
                        and (p.suffix in {'.py', '.sh', '.cpp', '.h', '.hpp', '.cmake'}
                             or p.name == 'CMakeLists.txt'))
    core = Path('/root/data/vllm-hust')
    frozen_core = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-hust')
    companion_core = []
    changed_core = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=core, text=True).splitlines()
    for filename in changed_core:
        source, actual = core / filename, frozen_core / filename
        if source.suffix != '.py' or not source.is_file():
            continue
        selected.add(source)
        matches = actual.is_file() and digest(source) == digest(actual)
        if actual.is_file():
            selected.add(actual)
        companion_core.append(dict(path=filename, working_source=str(source),
                                   frozen_runtime_source=str(actual), bytes_equal=matches))
    previous = DEST / 'SOURCE_MANIFEST.json'
    if previous.exists():
        for entry in json.loads(previous.read_text())['files']:
            if Path(entry['source']) not in selected:
                old = REPO / entry['saved']
                if old.exists():
                    if digest(old) != entry['sha256']:
                        raise RuntimeError(f'Previous saved file changed: {old}')
                    old.unlink()
    entries = []
    for source in sorted(selected):
        if source.is_symlink() or source.stat().st_size > 3 * 1024**2:
            raise RuntimeError(f'Unexpected source entry: {source}')
        target = DEST / 'snapshot' / source.relative_to('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entries.append(dict(source=str(source), saved=str(target.relative_to(REPO)),
                            sha256=digest(source), bytes=source.stat().st_size))
    manifest = dict(schema='SpecSLO-Tree-source-snapshot-v1',
        saved_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        upstream_branch_base=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        local_plugin_base=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=plugin, text=True).strip(),
        core_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd='/root/data/vllm-hust', text=True).strip(),
        companion_core_sources=companion_core,
        files=entries, frozen_entries=freezes, omitted_evidence=omitted,
        models_included=False, datasets_included=False, binary_caches_included=False,
        byte_exact_original_experiment_backup=False,
        original_absolute_paths_preserved=True)
    (DEST / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(files=len(entries), source_bytes=sum(v['bytes'] for v in entries),
                         omitted_evidence=len(omitted)), ensure_ascii=False))


if __name__ == '__main__':
    main()
