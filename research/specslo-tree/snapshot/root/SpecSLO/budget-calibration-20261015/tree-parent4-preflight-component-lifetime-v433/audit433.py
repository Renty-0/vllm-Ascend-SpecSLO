"""Full430 audit plus original preflight shadow phase covering every component."""
import json
from pathlib import Path
import audit430 as original


def audit(directory,selected_windows):
    value=original.audit(directory,selected_windows)
    comparisons=[]
    for rank in range(4):
        report=json.loads((Path(directory)/f'rank{rank}-local-diagnostic.json').read_text())
        row=report['ROOT433_preflight_component_lifetime'];state=row['after']
        if not(row['original218_preflight_source_unchanged']is True
            and row['original_close_called_once_after_all75']is True
            and row['all_original_and_new_component_preflights_compared']is True
            and row['production_generation_guard_preserved']is True
            and row['Native_flags_assigned']is False
            and row['before']['shadow']is True and row['before']['qualified']is False
            and state['shadow']is False and state['qualified']is True
            and state['shadow_pairs']>row['before']['shadow_pairs']):
            raise RuntimeError('ROOT433 genuine complete component comparisons and original close required')
        final=report['ROOT218_slot_preflight']
        if not(all(final[k]==state[k]for k in('shadow','qualified','shadow_pairs','shadow_target_pairs','shadow_draft_pairs','actual_target_dtype_counts'))):
            raise RuntimeError('ROOT433 no component comparison or flag change during performance')
        comparisons.append(dict(rank=rank,actual_component_comparisons=state['shadow_pairs']))
    return dict(value,ROOT433_complete_component_preflight_comparisons=comparisons,
        ROOT433_actual_production_generation_accounting_guard_preserved=True)
