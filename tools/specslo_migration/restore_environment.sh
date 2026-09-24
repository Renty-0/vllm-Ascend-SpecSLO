#!/usr/bin/env bash
set -euo pipefail

manifest_dir=${1:?usage: restore_environment.sh <manifest-directory> [repo-directory]}
repo_dir=${2:-/root/data/vllm-ascend-hust}

if [[ ! -d "$repo_dir/.git" ]]; then
    git clone "$manifest_dir/vllm-ascend-specslo.bundle" "$repo_dir"
fi

if ! /root/miniconda3/bin/conda env list | grep -q '^vllm-hust-dev[[:space:]]'; then
    /root/miniconda3/bin/conda env create \
        -n vllm-hust-dev \
        -f "$manifest_dir/vllm-hust-dev.environment.yml"
fi

echo "Repository and Conda metadata restored."
echo "CANN/driver must match the versions recorded in the manifest before NPU tests."
echo "Restore /data/shared-models and experiment artifacts, then rebuild the custom extension/vendor."
