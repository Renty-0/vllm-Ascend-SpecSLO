"""Prepare a separate full candidate using305; active299 stays frozen."""
import json
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
OLD = B / 'tree-capacity-request-risk-span-full-v299'
P = B / 'tree-capacity-risk-single-prepare-full-v307'


def main():
    P.mkdir(exist_ok=False)
    for source in OLD.iterdir():
        if not source.is_file() or (source.suffix not in {'.py', '.sh'} and source.name != 'FIXTURE299.json'):
            continue
        target = P / source.name.replace('299', '307')
        if source.suffix == '.json':
            target.write_bytes(source.read_bytes())
        else:
            target.write_text(source.read_text().replace(OLD.name, P.name).replace('299', '307'))
    for name in ('check307.py', 'runtime307.py'):
        path = P / name
        source = path.read_text()
        anchor = 'B = P.parent\n'
        assert source.count(anchor) == 1
        paths = [str(B / 'tree-single-validated-prefix-prepare-host-v305'),
                 str(B / 'tree-single-validated-prefix-descriptors-host-v304')]
        path.write_text(source.replace(anchor, anchor + 'sys.path.extend(' + repr(paths) + ')\n'))
    path = P / 'startup307.sh'
    source = path.read_text()
    assert source.count('export PYTHONPATH=') == 1
    path.write_text(source.replace('export PYTHONPATH=', 'export PYTHONPATH=' +
        str(B / 'tree-single-validated-prefix-prepare-host-v305') + ':' +
        str(B / 'tree-single-validated-prefix-descriptors-host-v304') + ':'))
    (P / 'static_state307.py').write_text(
        '"""HOST305 prepare optimization; fresh full-instance model checks required."""\n'
        'from static_state305 import ensure_zero,bind_checked_prepare,install,require_copy_bindings\n')
    path = P / 'extension307.py'
    source = path.read_text()
    assert source.count('engine._branch_tail_state.prefix296_copies.width') == 1
    path.write_text(source.replace('engine._branch_tail_state.prefix296_copies.width',
                                   'engine._branch_tail_state.prefix305_copies.width'))
    review = dict(status='SOURCE_ONLY307_fast_prepare_not_frozen_not_NPU_run',
        only_serving_delta='Same capacity/risk policy and raw operators; original map validation once, direct span/alias descriptors and no unused prefix staging',
        kernels_and_attention_from_frozen296=True, models_and_sampler_unchanged=True,
        active299_sources_edited=False, new_model_qualification_inherited=False,
        original_startup_408target45draft20followup463ownAR_required=True,
        prerequisite='Finish299 andinspectactualfailures orperformance before scheduling307',
        NPU_used=False, Goodput=None, adopted=False)
    (P / 'SOURCE_ONLY307_REVIEW.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps(review))


if __name__ == '__main__':
    main()
