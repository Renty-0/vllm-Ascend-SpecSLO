#!/usr/bin/env bash
set -euo pipefail

destination=${1:?usage: restore.sh <destination-root>}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

[[ -d "$destination" ]] || mkdir -p "$destination"
cd "$script_dir"
sha256sum -c SHA256SUMS

restore_layer() {
    local prefix=$1
    local parts=()
    local expected

    shopt -s nullglob
    parts=("$prefix"[0-9][0-9][0-9])
    shopt -u nullglob
    (( ${#parts[@]} > 0 )) || {
        echo "no archive parts found for $prefix" >&2
        exit 1
    }
    for ((index = 0; index < ${#parts[@]}; index++)); do
        expected=$(printf '%s%03d' "$prefix" "$index")
        [[ "${parts[$index]}" == "$expected" ]] || {
            echo "archive parts are not contiguous for $prefix" >&2
            exit 1
        }
    done
    cat "${parts[@]}" | zstd -t
    cat "${parts[@]}" | zstd -dc | tar -tf - >/dev/null
    cat "${parts[@]}" | zstd -dc \
        | tar --numeric-owner --acls --xattrs -C "$destination" -xf -
}

restore_layer "$script_dir/specslo-portable-evidence-20260930.tar.zst.part-"
restore_layer "$script_dir/specslo-compiled-runtime-20260930.tar.zst.part-"

echo "restored portable evidence and compiled runtime under $destination"
