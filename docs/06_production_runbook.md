# Production Runbook

Vertical Brain's production storage backend is SQLite. The JSON backend is for
development and debugging only.

## Health Check

Run integrity checks before and after maintenance:

```bash
vb --data-dir /path/to/brain doctor
```

For machine-readable output:

```bash
vb --data-dir /path/to/brain doctor --json
```

## Backup

Create a consistent SQLite backup with the SQLite backup API:

```bash
vb --data-dir /path/to/brain backup /path/to/backups/vertical_brain.sqlite
```

By default the command refuses to overwrite an existing backup. Use
`--overwrite` only when replacing the target is intentional.

## WAL Checkpoint

SQLite runs in WAL mode. After bulk writes or before external file-level backup,
checkpoint the WAL:

```bash
vb --data-dir /path/to/brain checkpoint --mode truncate
```

Use `--json` when automation needs the `busy`, `log`, and `checkpointed`
counters.

## Vacuum

`mark_stale` and `supersede_chunk` make old facts invisible to normal reads and
search immediately, but keep them on disk for audit-friendly retention. Use
`vacuum` for Databricks-style physical cleanup after the retention window:

```bash
vb --data-dir /path/to/brain vacuum --retention-hours 168
```

The command is a dry run by default. To apply, create a backup and pass
`--apply`:

```bash
vb --data-dir /path/to/brain vacuum \
  --apply \
  --backup /path/to/backups/before-vacuum.sqlite
```

Vacuum deletes eligible `stale`, `superseded`, `legacy`, and `contradicted`
chunks, prunes orphan embedding vectors while preserving active chunk and Gold
aspect vectors, removes empty namespace nodes, drops the deleted chunks from the
FTS index, and checkpoints the WAL. Retention below 168 hours
requires `--force`. Use `--reclaim-space` when you also want SQLite to run a
full `VACUUM` and shrink the database file.

## Restore

Stop writers, copy the backup file into the data directory as
`vertical_brain.sqlite`, then run:

```bash
vb --data-dir /path/to/brain doctor
```

If semantic search is used with a different embedding model after restore, run
`vb reindex --embedding-url <endpoint> --embedding-model <model>` before serving
semantic queries — it rebuilds the vector cache under the new model.
