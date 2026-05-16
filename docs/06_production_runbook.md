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

## Restore

Stop writers, copy the backup file into the data directory as
`vertical_brain.sqlite`, then run:

```bash
vb --data-dir /path/to/brain doctor
```

If semantic search is used with a different embedding model after restore, run
the reindexing flow before serving semantic queries.
