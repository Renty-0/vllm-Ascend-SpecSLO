# Codex conversations backup — 2026-09-24

This orphan branch stores a point-in-time backup of the Codex conversation
history from the SpecSLO development container.

## Contents

- 296 JSONL files from `sessions/`, `archived_sessions/`, and
  `session_index.jsonl`;
- consistent SQLite online backups of `thread_history_1.sqlite`,
  `state_5.sqlite`, and `goals_1.sqlite`;
- a per-source-file checksum manifest inside the archive;
- 12 numbered zstd parts, totaling 1,044,985,195 bytes before Git storage.

The uncompressed snapshot is 6,965,411,840 bytes. The SQLite snapshots passed
`PRAGMA integrity_check`; the concatenated zstd stream and tar index were also
read successfully before this branch was created.

## Sensitive-data boundary

The branch intentionally excludes `auth.json`, `.env`, `config.toml`, SSH
keys, installation identifiers, operational logs, queues, caches, plugins,
skills, shell snapshots, IPC sockets, lock files, and SQLite WAL/SHM files.
An exact scan found none of the current Codex authentication tokens and no PEM
private-key marker in the snapshot.

Conversation rollouts still contain prompts, assistant messages, tool calls,
terminal output, and environment details. Keep this repository and branch
private and do not open a public pull request from it.

## Verify and restore

```bash
git clone --single-branch \
  --branch codex-conversations-backup-20260924 \
  git@github.com:Renty-0/vllm-Ascend-SpecSLO.git codex-conversations-backup
cd codex-conversations-backup
sha256sum -c SHA256SUMS
./restore.sh /tmp/codex-conversations-restore
```

The destination argument is mandatory. Restoring to a temporary directory is
recommended first; inspect `DESTINATION/root/.codex` before copying anything
over a newer Codex installation.
