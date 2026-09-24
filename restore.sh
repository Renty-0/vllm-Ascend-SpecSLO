#!/usr/bin/env bash
set -euo pipefail

destination=${1:?usage: restore.sh <destination-root>}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
prefix="$script_dir/codex-conversations-20260924.tar.zst.part-"

[[ -d "$destination" ]] || mkdir -p "$destination"

cd "$script_dir"
sha256sum -c SHA256SUMS

shopt -s nullglob
parts=("$prefix"[0-9][0-9][0-9])
shopt -u nullglob
(( ${#parts[@]} > 0 )) || {
    echo "no archive parts found" >&2
    exit 1
}

for ((index = 0; index < ${#parts[@]}; index++)); do
    expected=$(printf '%s%03d' "$prefix" "$index")
    if [[ "${parts[$index]}" != "$expected" ]]; then
        echo "archive parts are not contiguous" >&2
        exit 1
    fi
done

cat "${parts[@]}" | zstd -t
cat "${parts[@]}" | zstd -dc | tar -tf - >/dev/null
cat "${parts[@]}" | zstd -dc \
    | tar --numeric-owner --acls --xattrs -C "$destination" -xf -

echo "restored conversation snapshot under $destination/root/.codex"
