#!/usr/bin/env bash
set -euo pipefail

manifest_dir=${1:?usage: verify_manifest.sh <manifest-directory>}
cd "$manifest_dir"
sha256sum -c MANIFEST.sha256

if [[ -f vllm-ascend-specslo.bundle ]]; then
    git bundle verify vllm-ascend-specslo.bundle
fi
