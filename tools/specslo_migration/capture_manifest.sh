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

capture_repository() {
    local name=$1
    local path=$2
    local destination="$output_dir/repositories/$name"
    [[ -d "$path/.git" || -f "$path/.git" ]] || return 0
    mkdir -p "$destination"
    git -C "$path" status --short > "$destination/status.txt"
    git -C "$path" remote -v > "$destination/remotes.txt"
    git -C "$path" log --oneline --decorate -50 > "$destination/log.txt"
    git -C "$path" rev-parse HEAD > "$destination/head.txt"
    git -C "$path" bundle create "$destination/repository.bundle" --all
    git -C "$path" diff --binary > "$destination/unstaged.patch"
    git -C "$path" diff --cached --binary > "$destination/staged.patch"
    git -C "$path" ls-files --others --exclude-standard \
        > "$destination/untracked-files.txt"
}

capture_repository vllm-ascend-hust "$repo"
capture_repository vllm-hust /root/data/vllm-hust
capture_repository vllm-hust-dev-hub /root/data/vllm-hust-dev-hub

/root/miniconda3/bin/conda env export -n vllm-hust-dev --no-builds \
    > "$output_dir/vllm-hust-dev.environment.yml"
/root/miniconda3/bin/conda list -n vllm-hust-dev --explicit \
    > "$output_dir/vllm-hust-dev.conda-explicit.txt"
"$python_bin" -m pip freeze --all > "$output_dir/vllm-hust-dev.pip-freeze.txt"
"$python_bin" -VV > "$output_dir/python-version.txt" 2>&1
"$python_bin" -m pip show torch torch-npu vllm vllm-ascend \
    > "$output_dir/runtime-packages.txt" 2>&1 || true

for version_file in \
    /usr/local/Ascend/ascend-toolkit/latest/version.cfg \
    /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascend_toolkit_install.info \
    /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info \
    /usr/local/Ascend/driver/version.info \
    /usr/local/Ascend/nnal/atb/latest/atb/version.info; do
    if [[ -f "$version_file" ]]; then
        safe_name=${version_file#/}
        safe_name=${safe_name//\//-}
        cp "$version_file" "$output_dir/$safe_name"
    fi
done

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
    find . -type f ! -name MANIFEST.sha256 -print0 \
        | sort -z \
        | xargs -0 -r sha256sum > MANIFEST.sha256
)

echo "migration manifest written to $output_dir"
