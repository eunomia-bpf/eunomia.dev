# Syndication ledger

Per-post record of where each article has been republished and whether the
copy carries `rel=canonical` back to eunomia.dev. Update this file whenever a
post is syndicated (manually or via the queue), and audit it before adding
anything to `posts_queue.txt`. Canonical must be set in the platform editor at
publish time (the automated pipeline does not send it).

Status legend: `ok` = live with canonical; `missing` = live WITHOUT canonical
(needs backfill); `queued` = scheduled; `-` = not published there.

| Post (docs/blog/posts/) | Platform | URL | Canonical | Date | Notes |
|---|---|---|---|---|---|
| _example: agentsight_paper.md_ | dev.to | _fill_ | missing | _fill_ | backfill canonical in dev.to editor |

## Backfill worklist (2026-07-16)

Historical dev.to/Medium copies were published without canonical (verified for
the AgentSight posts). Enumerate every existing copy on both platforms, add a
row above for each, set canonical in the platform editor, then flip the status
to `ok`. The syndication queue stays paused until this list is clean.
