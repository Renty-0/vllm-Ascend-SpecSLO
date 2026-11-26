#!/usr/bin/env bash
set -euo pipefail
umask 077

tag=${1:?usage: run_baseline_v021_inner.sh UNIQUE_TAG}
batch=${2:-64}
[[ $batch == 8 || $batch == 16 || $batch == 32 || $batch == 64 ]] || exit 2
repro=${LLAMA_REPRO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
run=$repro/baseline-results/runs/official-v021-llama70b-b${batch}-${tag}
[[ ! -e $run ]]
mkdir -p "$run"
cd /root

# There is no qualified Llama-70B MC2 profile in this environment.
export VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=0 VLLM_ASCEND_ENABLE_FUSED_MC2=0
export VLLM_ASCEND_ENABLE_NZ=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT="$run/compile-cache"
export TASK_QUEUE_ENABLE=1 HCCL_DETERMINISTIC=true HCCL_CONNECT_TIMEOUT=900
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
unset HCCL_OP_EXPANSION_MODE ASCEND_CUSTOM_OPP_PATH VLLM_ASCEND_ACLGRAPH_BENCHMARK_JSONL

npu-smi info > "$run/npu-before.txt"
python -B - "$run" "$batch" <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import vllm
import vllm_ascend

run = Path(sys.argv[1])
batch = int(sys.argv[2])
root = Path('/root/SpecSLO/official-v021-baseline-20261019')
git = lambda path: subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
data = dict(created_unix=time.time(), batch_size=batch, python=sys.version,
            packages={n: importlib.metadata.version(n) for n in
                      ['torch', 'torch-npu', 'vllm', 'vllm-ascend', 'transformers', 'triton-ascend']},
            ascend_commit=git(root / 'vllm-ascend'), vllm_commit=git(root / 'vllm'),
            vllm_source=vllm.__file__, vllm_ascend_source=vllm_ascend.__file__,
            physical_devices=[int(x) for x in os.environ['SPECSLO_VISIBLE_DEVICES'].split(',')],
            model='/data/shared_upload/models/Meta-Llama-3.1-70B-Instruct',
            request_manifest='/root/data/specslo-workloads/eurosys27-rps4-seed0-author-slo40-50-150/rps4.0_mix0.6_0.2_0.2_seed0.jsonl',
            environment={k: os.environ.get(k) for k in
                         ['ASCEND_RT_VISIBLE_DEVICES', 'PYTHONPATH', 'LD_LIBRARY_PATH',
                          'VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE', 'VLLM_ASCEND_ENABLE_FUSED_MC2',
                          'VLLM_ASCEND_ENABLE_NZ', 'HCCL_OP_EXPANSION_MODE', 'TASK_QUEUE_ENABLE',
                          'HCCL_DETERMINISTIC', 'HCCL_CONNECT_TIMEOUT', 'VLLM_CACHE_ROOT']},
            inference_source_modified=False)
assert data['ascend_commit'] == '80610e4438dba05011b05f89fc45d91e96992671'
assert data['vllm_commit'] == 'ad7125a431e176d4161099480a66f0169609a690'
(run / 'launch-environment.json').write_text(json.dumps(data, indent=2) + '\n')
PY

exec timeout --signal=TERM --kill-after=30 1800 python -B -u \
  /root/SpecSLO/official-v021-baseline-20261112/harness/examples/benchmark_nano_pearl_target_only.py \
  --model /data/shared_upload/models/Meta-Llama-3.1-70B-Instruct --tensor-parallel-size 4 --dtype bfloat16 \
  --max-model-len 4096 --max-num-seqs "$batch" --gpu-memory-utilization 0.8 \
  --max-num-batched-tokens 640 --max-cudagraph-capture-size 640 \
  --cudagraph-capture-sizes 1,2,4,8,16,24,32,40,48,56,64,128,192,256,320,384,448,512,640 \
  --cudagraph-copy-inputs --batch-sizes "$batch" --num-prompts 463 --max-tokens 256 \
  --warmup-prompts "$batch" --warmup-max-tokens 256 --warmup-runs 1 \
  --request-manifest /root/data/specslo-workloads/eurosys27-rps4-seed0-author-slo40-50-150/rps4.0_mix0.6_0.2_0.2_seed0.jsonl \
  --online-arrivals --prefill-coalesce-min-requests 1 --prefill-coalesce-max-wait-ms 0 \
  --output-json "$run/result.json" > "$run/run.log" 2>&1
