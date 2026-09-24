#!/usr/bin/env bash
set -euo pipefail

archive_dir=${1:?usage: verify_portable_archives.sh <archive-directory>}
cd "$archive_dir"
sha256sum -c SHA256SUMS

shopt -s nullglob
first_parts=(*.tar.zst.part-000)
shopt -u nullglob

for first_part in "${first_parts[@]}"; do
    [[ -e "$first_part" ]] || continue
    prefix=${first_part%000}
    shopt -s nullglob
    parts=("${prefix}"[0-9][0-9][0-9])
    shopt -u nullglob
    for ((index = 0; index < ${#parts[@]}; index++)); do
        expected=$(printf '%s%03d' "$prefix" "$index")
        if [[ "${parts[$index]}" != "$expected" ]]; then
            echo "non-contiguous archive chunks for $prefix" >&2
            exit 1
        fi
    done
    echo "testing ${prefix}[000-$(printf '%03d' "$(( ${#parts[@]} - 1 ))")]"
    cat "${parts[@]}" | zstd -t
    cat "${parts[@]}" | zstd -dc | tar -tf - >/dev/null
done
