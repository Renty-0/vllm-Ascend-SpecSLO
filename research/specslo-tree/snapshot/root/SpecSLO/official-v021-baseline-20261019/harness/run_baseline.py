"""Same original arrival driver with untimed official-source/model evidence."""
import json
import os
import time
from pathlib import Path

import vllm
import vllm_ascend
from examples import benchmark_nano_pearl_target_only as benchmark

ROOT = Path('/root/SpecSLO/official-v021-baseline-20261019')


def main():
    if not Path(vllm.__file__).resolve().is_relative_to(ROOT / 'vllm'):
        raise RuntimeError('Official v0.21 core source required')
    if not Path(vllm_ascend.__file__).resolve().is_relative_to(ROOT / 'vllm-ascend'):
        raise RuntimeError('Official 80610e44 plugin source required')
    output_dir = Path(os.environ['VLLM_CACHE_ROOT']).parent
    original_llm = vllm.LLM
    original_online = benchmark._run_online_arrivals

    def measured_online(*args, **kwargs):
        marker = output_dir / 'measurement-markers.json'
        interval = dict(start_unix=time.time(), end_unix=None)
        marker.write_text(json.dumps(interval) + '\n')
        try:
            return original_online(*args, **kwargs)
        finally:
            interval['end_unix'] = time.time()
            marker.write_text(json.dumps(interval) + '\n')

    benchmark._run_online_arrivals = measured_online

    class EvidenceLLM(original_llm):
        def __init__(self, *args, **kwargs):
            kwargs['worker_extension_cls'] = 'tp4_model_evidence.TP4EvidenceExtension'
            super().__init__(*args, **kwargs)
            evidence = self.collective_rpc('specslo_tp4_model_evidence', timeout=120)
            (output_dir / 'model-evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
            if len(evidence) != 4 or {row['rank'] for row in evidence} != {0, 1, 2, 3}:
                raise RuntimeError('All four actual TP4 workers required')
            for row in evidence:
                if not Path(row['vllm_source']).resolve().is_relative_to(ROOT / 'vllm'):
                    raise RuntimeError('Worker core must use old official source')
                if not Path(row['vllm_ascend_source']).resolve().is_relative_to(ROOT / 'vllm-ascend'):
                    raise RuntimeError('Worker plugin must use old official source')
                if not row['enabled'] or row['selected_modules'] != 128 or row['quantization'] is not None:
                    raise RuntimeError('Unquantized actual MC2 selection required')

    vllm.LLM = EvidenceLLM
    benchmark.main()


if __name__ == '__main__':
    main()
