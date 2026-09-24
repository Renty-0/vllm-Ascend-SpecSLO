#!/usr/bin/env bash
set -euo pipefail

archive_dir=${1:?usage: restore_portable_archive.sh <archive-directory> <archive-name> [destination-root]}
archive_name=${2:?usage: restore_portable_archive.sh <archive-directory> <archive-name> [destination-root]}
destination=${3:-/}
prefix="$archive_dir/$archive_name.tar.zst.part-"

[[ -d "$destination" ]] || {
    echo "destination root does not exist: $destination" >&2
    exit 1
}
[[ -f "$archive_dir/SHA256SUMS" ]] || {
    echo "missing checksum file: $archive_dir/SHA256SUMS" >&2
    exit 1
}

shopt -s nullglob
parts=("$prefix"[0-9][0-9][0-9])
shopt -u nullglob
if (( ${#parts[@]} == 0 )); then
    echo "no chunks found for $archive_name" >&2
    exit 1
fi

for ((index = 0; index < ${#parts[@]}; index++)); do
    expected=$(printf '%s%03d' "$prefix" "$index")
    if [[ "${parts[$index]}" != "$expected" ]]; then
        echo "non-contiguous archive chunks for $archive_name" >&2
        exit 1
    fi
done

for part in "${parts[@]}"; do
    checksum_name="./${part##*/}"
    checksum_line=$(awk -v name="$checksum_name" '$2 == name { print; found = 1 } END { if (!found) exit 1 }' \
        "$archive_dir/SHA256SUMS") || {
        echo "missing checksum for ${part##*/}" >&2
        exit 1
    }
    (cd "$archive_dir" && printf '%s\n' "$checksum_line" | sha256sum -c -)
done

cat "${parts[@]}" | zstd -dc | tar -tf - >/dev/null
cat "${parts[@]}" | zstd -dc \
    | tar --acls --xattrs --numeric-owner -C "$destination" -xf -
