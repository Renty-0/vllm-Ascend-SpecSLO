#!/usr/bin/env bash
set -euo pipefail

output_dir=${1:-/root/data/specslo-migration-manifest}
repo=${SPECSLO_REPO:-/root/data/vllm-ascend-hust}
python_bin=${SPECSLO_PYTHON:-/root/miniconda3/envs/vllm-hust-dev/bin/python}

mkdir -p "$output_dir"

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$output_dir/captured_at_utc.txt"
uname -a > "$output_dir/uname.txt"
cp /etc/os-release "$output_dir/os-release"
df -hT > "$output_dir/filesystems.txt"
findmnt -o TARGET,SOURCE,FSTYPE,OPTIONS > "$output_dir/mounts.txt"

if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info > "$output_dir/npu-smi-info.txt" 2>&1 || true
fi

git -C "$repo" status --short > "$output_dir/repository-status.txt"
git -C "$repo" remote -v > "$output_dir/repository-remotes.txt"
git -C "$repo" log --oneline --decorate -50 > "$output_dir/repository-log.txt"
git -C "$repo" rev-parse HEAD > "$output_dir/repository-head.txt"
git -C "$repo" bundle create "$output_dir/vllm-ascend-specslo.bundle" --all
git -C "$repo" diff --binary > "$output_dir/unstaged.patch"
git -C "$repo" diff --cached --binary > "$output_dir/staged.patch"

/root/miniconda3/bin/conda env export -n vllm-hust-dev --no-builds \
    > "$output_dir/vllm-hust-dev.environment.yml"
/root/miniconda3/bin/conda list -n vllm-hust-dev --explicit \
    > "$output_dir/vllm-hust-dev.conda-explicit.txt"
"$python_bin" -m pip freeze --all > "$output_dir/vllm-hust-dev.pip-freeze.txt"
"$python_bin" -VV > "$output_dir/python-version.txt" 2>&1

# File-size manifests are intentionally cheap enough to refresh repeatedly.
# Large model hashes are generated separately because reading 171+ GB on every
# capture would interfere with active experiments.
for root in /root/data /data/shared-models /data/datasets; do
    if [[ -d "$root" ]]; then
        name=${root#/}
        name=${name//\//-}
        find "$root" -xdev -type f -printf '%s\t%T@\t%p\n' \
            | sort -k3 > "$output_dir/$name.files.tsv"
    fi
done

find /root/data /data/shared-models /data/datasets -xdev -type f \
    \( -name '*.json' -o -name '*.md' -o -name '*.yaml' -o -name '*.yml' \
       -o -name '*.toml' -o -name '*.sh' -o -name '*.py' -o -name '*.run' \) \
    -print0 2>/dev/null \
    | sort -z \
    | xargs -0 -r sha256sum > "$output_dir/metadata.sha256"

(
    cd "$output_dir"
    sha256sum -- * > MANIFEST.sha256
)

echo "migration manifest written to $output_dir"
