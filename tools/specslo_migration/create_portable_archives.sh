#!/usr/bin/env bash
set -euo pipefail

output_dir=${1:?usage: create_portable_archives.sh <output-directory> [chunk-size]}
chunk_size=${2:-4G}
compression_level=${SPECSLO_ZSTD_LEVEL:-1}
manifest_dir=${SPECSLO_MANIFEST_DIR:-}

umask 077
mkdir -p "$output_dir"
output_dir=$(realpath -m -- "$output_dir")
chmod 700 "$output_dir"

private_paths=(/root/.ssh /root/.gitconfig /root/.bashrc /root/.bash_profile)
[[ -d /root/.config ]] && private_paths+=(/root/.config)

source_roots=(/root/data /root/miniconda3 /data/datasets)
[[ "${SPECSLO_INCLUDE_CODEX:-1}" == 1 ]] && source_roots+=(/root/.codex)
if [[ "${SPECSLO_INCLUDE_PRIVATE_CONFIG:-1}" == 1 ]]; then
    source_roots+=("${private_paths[@]}")
fi
if [[ "${SPECSLO_INCLUDE_KEY_MODELS:-0}" == 1 ]]; then
    source_roots+=(
        /data/shared-models/Qwen3-32B
        /data/shared-models/Qwen3-0.6B
        /data/shared-models/Qwen2.5-14B-Instruct
        /data/shared-models/Qwen2.5-0.5B-Instruct
    )
fi
[[ -n "$manifest_dir" ]] && source_roots+=("$manifest_dir")

for source_root in "${source_roots[@]}"; do
    [[ -e "$source_root" ]] || {
        echo "archive source does not exist: $source_root" >&2
        exit 1
    }
    source_root=$(realpath -m -- "$source_root")
    if [[ -d "$source_root" ]] \
        && { [[ "$output_dir" == "$source_root" ]] \
             || [[ "$output_dir" == "$source_root/"* ]]; }; then
        echo "output directory must not be inside archive source: $source_root" >&2
        exit 1
    fi
done

active_partial_dir=
index_tmp="$output_dir/.ARCHIVE_INDEX.tsv.$$"

cleanup() {
    if [[ -n "$active_partial_dir" && -d "$active_partial_dir" ]]; then
        rm -rf -- "$active_partial_dir"
    fi
    rm -f -- "$index_tmp"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

printf 'archive_name\tsensitive\tcaptured_at_utc\tsource_paths\n' > "$index_tmp"

create_archive() {
    local name=$1
    local sensitive=$2
    shift 2
    local prefix="$output_dir/$name.tar.zst.part-"
    local source_list=
    local source_path
    local tar_paths=()
    local existing_parts=()
    local generated_parts=()

    shopt -s nullglob
    existing_parts=("$prefix"[0-9][0-9][0-9])
    shopt -u nullglob
    if (( ${#existing_parts[@]} > 0 )); then
        echo "$name already has chunks; refusing to overwrite" >&2
        return 1
    fi

    for source_path in "$@"; do
        source_path=$(realpath -m -- "$source_path")
        tar_paths+=("${source_path#/}")
        if [[ -n "$source_list" ]]; then
            source_list+=";"
        fi
        source_list+="$source_path"
    done

    active_partial_dir=$(mktemp -d "$output_dir/.partial.${name}.XXXXXX")
    local partial_prefix="$active_partial_dir/$name.tar.zst.part-"
    echo "creating $name from: $source_list"
    tar --acls --xattrs --numeric-owner --sparse --one-file-system \
        --exclude=root/.codex/ipc/ipc.sock \
        -C / -cf - "${tar_paths[@]}" \
        | zstd -T0 -"$compression_level" \
        | split -b "$chunk_size" -d -a 3 - "$partial_prefix"

    shopt -s nullglob
    generated_parts=("$partial_prefix"[0-9][0-9][0-9])
    shopt -u nullglob
    if (( ${#generated_parts[@]} == 0 )); then
        echo "archive produced no chunks: $name" >&2
        return 1
    fi
    for source_path in "${generated_parts[@]}"; do
        mv -- "$source_path" "$output_dir/${source_path##*/}"
    done
    rmdir -- "$active_partial_dir"
    active_partial_dir=
    printf '%s\t%s\t%s\t%s\n' \
        "$name" "$sensitive" "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        "$source_list" >> "$index_tmp"
}

create_archive root-data no /root/data
create_archive miniconda3 no /root/miniconda3
create_archive project-datasets no /data/datasets

if [[ "${SPECSLO_INCLUDE_CODEX:-1}" == 1 ]]; then
    create_archive codex-state yes /root/.codex
fi

# Contains SSH private keys and application configuration. Chunks are mode
# 0600, but they must still travel and remain at rest only through a protected
# private channel.
if [[ "${SPECSLO_INCLUDE_PRIVATE_CONFIG:-1}" == 1 ]]; then
    create_archive private-config yes "${private_paths[@]}"
fi

if [[ "${SPECSLO_INCLUDE_KEY_MODELS:-0}" == 1 ]]; then
    create_archive model-qwen3-32b no /data/shared-models/Qwen3-32B
    create_archive model-qwen3-06b no /data/shared-models/Qwen3-0.6B
    create_archive model-qwen25-14b no /data/shared-models/Qwen2.5-14B-Instruct
    create_archive model-qwen25-05b no /data/shared-models/Qwen2.5-0.5B-Instruct
fi

if [[ -n "$manifest_dir" ]]; then
    create_archive migration-manifest no "$manifest_dir"
fi

mv -- "$index_tmp" "$output_dir/ARCHIVE_INDEX.tsv"
(
    cd "$output_dir"
    find . -maxdepth 1 -type f \
        \( -name '*.tar.zst.part-[0-9][0-9][0-9]' \
           -o -name ARCHIVE_INDEX.tsv \) -print0 \
        | sort -z \
        | xargs -0 -r sha256sum > SHA256SUMS
)

trap - EXIT INT TERM HUP
echo "portable archives written to $output_dir"
