"""Explicit local-first entry and readonly module-origin check before models."""
import importlib
from pathlib import Path
import sys
P=Path(__file__).resolve().parent
KEYS=('scope_v146','dual_bank_v146','draft_v146','draft_observer_v146','target_v146','actor_v146','audit_v146','full_process_v146','variable_actual_gate_v146','no_mixed_prefill_v143','normal_admission_v140','gain_state_v140','exception_evidence_v140','config_snapshot_v138','cpu_default_device_cleanup_host','tree_entry','budget_policy','root_seed','fast_tree_draft_v5','chain_metadata','kv_ledger','pa_repair',
 'parallel_steps','tail_hidden_observer','target_layout','host_target_metadata','host_tree_transport',
 'tree_target_replay','prefill_exact','prefill_inventory','prefill_arena','target_inventory','draft_inventory',
 'finite_inventory_plan','mc2_inventory_contract','local_mixed_qualification','independent_history_v75',
 'normal_draft_actor_v75','native_boundaries','original_tree_cycle_v75','mixed_target_reference_v75',
 'mixed_target','live_native_boundaries','live_runtime_hooks','live_decode_patch','production_probe_v101',
 'poisson_coverage_probe','live_poisson_probe','shape_scope_v101','serialized_structure_evidence_v129',
 'split_budget_v123','split_budget_v132','split_shape_v132','runtime_adapter_v132','candidate_controls_v132')

def local_first():
    sys.path[:]=[str(P),*(value for value in sys.path if Path(value or '.').resolve()!=P)]
    rows={}
    for name in KEYS:
        if name in sys.modules:
            module=sys.modules[name];origin=Path(getattr(module,'__file__','/')).resolve()
            if origin!=P/(name+'.py'):raise RuntimeError('V132 actual local helper origin differs: '+name+': '+str(origin))
            rows[name]=str(origin)
    return rows

def require_loaded_origins():
    local_first()
    for name in KEYS:importlib.import_module(name)
    rows=local_first()
    if set(rows)!=set(KEYS):raise RuntimeError('V132 actual constructor helper source inventory incomplete')
    return rows
