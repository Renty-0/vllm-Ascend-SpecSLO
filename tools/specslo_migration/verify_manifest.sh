#!/usr/bin/env bash
set -euo pipefail

manifest_dir=${1:?usage: verify_manifest.sh <manifest-directory>}
cd "$manifest_dir"
sha256sum -c MANIFEST.sha256

verification_repo=$(mktemp -d)
trap 'rm -rf "$verification_repo"' EXIT
git init --bare "$verification_repo" >/dev/null
while IFS= read -r -d '' bundle; do
    git -C "$verification_repo" bundle verify "$PWD/$bundle"
done < <(find repositories -type f -name repository.bundle -print0 | sort -z)
