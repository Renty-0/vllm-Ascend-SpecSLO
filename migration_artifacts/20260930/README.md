# SpecSLO portable evidence and compiled-runtime snapshot

This directory is the GitHub-sized recovery layer captured on 2026-09-30.
It deliberately separates reproducibility evidence from the much larger model
weights, tensor captures, raw CANN profiles, Conda tree, and system CANN
installation listed in `docs/migration/20260930/README_zh.md`.

## Contents

- `specslo-portable-evidence-20260930.tar.zst.part-*`: 21,937 files,
  1,704,355,840 bytes before compression. It includes all benchmark
  `result.json` files, all JSON files smaller than 5 MiB, reports, Markdown,
  CSV/TSV, launch scripts, environment snapshots, checksum markers, the
  SpecSLO workloads, and `/data/datasets`.
- `SOURCE_SHA256SUMS.zst`: hashes of every file in the evidence archive.
- `specslo-compiled-runtime-20260930.tar.zst.part-*`: the installed SpecSLO
  MC2, custom-transformer and batch-invariant CANN vendors plus the built
  `vllm_ascend` extension libraries (703 files, 232,284,160 bytes before
  compression).
- `COMPILED_RUNTIME_SOURCE_SHA256SUMS.zst`: hashes for the compiled-runtime
  archive.
- `specslo-tp4-baseline-result-20260928.json.zst`: the complete 463-request
  TP4 target-only baseline result.  It is stored separately because the
  original file ended in `.json.tmp` and was consequently omitted by the
  first portable-evidence selection.  Its decompressed SHA256 is
  `2b957c1e59b1443e9d9cddd0bc2481a248d94db24808cbe06d590eb1bee198cd`.
- `SHA256SUMS`: hashes for all Git-stored compressed files.

The compiled runtime is an exact rollback aid, not a portable substitute for
matching CANN, driver, firmware, PyTorch and torch-npu ABIs. Rebuilding from
the committed source remains the preferred path on a new image.

Before packaging, all selected evidence files were scanned against the current
Codex authentication strings and proxy/environment values, plus PEM private-key
markers. All three scans returned zero matches. Benchmark variables such as
`linear_bonus_token` are experiment parameters, not authentication tokens.

## Verify and extract

```bash
sha256sum -c SHA256SUMS
./restore.sh /tmp/specslo-restore
cd /tmp/specslo-restore
zstd -dc /path/to/SOURCE_SHA256SUMS.zst | sha256sum -c -
zstd -dc /path/to/COMPILED_RUNTIME_SOURCE_SHA256SUMS.zst | sha256sum -c -
```

Extract to a temporary root first. Do not copy the archived CANN vendor or
extension into a live environment until its ABI has been checked.

Restore the full historical baseline evidence with:

```bash
zstd -dc specslo-tp4-baseline-result-20260928.json.zst \
  > /tmp/specslo-tp4-baseline-result-20260928.json
printf '%s  %s\n' \
  2b957c1e59b1443e9d9cddd0bc2481a248d94db24808cbe06d590eb1bee198cd \
  /tmp/specslo-tp4-baseline-result-20260928.json | sha256sum -c -
```
