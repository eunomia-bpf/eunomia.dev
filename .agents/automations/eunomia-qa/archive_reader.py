#!/usr/bin/env python3
"""Read-only Slack archive probe and snapshot helper.

Reads the community watchlist, verifies every archive invariant before the
first data read, and emits bounded diagnostics. Probe and snapshot stdout
contains only counts, booleans, and bounded reason codes: never raw message
text, identities, schema names, the DSN, SQL, or exception details.

Windows are computed from the Slack message ts column only; load or
ingestion time is never read.
"""

from __future__ import annotations

import os
import sys
import time
from decimal import Decimal
from pathlib import Path

DSN_ENV = "EUNOMIA_QA_ARCHIVE_DSN"
MAX_SNAPSHOT_BYTES = 120000
SNAPSHOT_WINDOW_HOURS = 168
PROBE_WINDOWS_HOURS = (24, 168)
REASON_CODES = frozenset(
    {
        "ok",
        "no_opt_in",
        "dsn_missing",
        "connect_failed",
        "readonly_unenforceable",
        "role_privileged",
        "write_grant_found",
        "schema_none",
        "schema_ambiguous",
        "channel_missing",
        "read_failed",
        "snapshot_too_large",
        "output_exists",
        "output_refused",
    }
)
EXIT_OK = 0
EXIT_INACCESSIBLE = 1
EXIT_USAGE = 2


class _InvariantFailure(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _quote_ident(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _close(conn) -> None:
    try:
        conn.close()
    except Exception:
        pass


def default_watchlist_path():
    return (
        Path(__file__).resolve().parents[3]
        / ".github"
        / "publisher"
        / "media"
        / "community-watchlist.yaml"
    )


def load_slack_sources(watchlist_path) -> list:
    import yaml

    with open(watchlist_path, "r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    sources = []
    for entry in (doc or {}).get("sources") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("surface") or "").strip() != "Slack":
            continue
        if entry.get("archive_opt_in") is not True:
            continue
        team = entry.get("workspace")
        if not isinstance(team, str) or not team.strip():
            continue
        raw_channels = entry.get("channels")
        if not isinstance(raw_channels, list):
            raw_channels = []
        channels = [c for c in raw_channels if isinstance(c, str) and c]
        if not channels:
            continue
        sources.append({"team": team, "channels": channels})
    return sources


def _connect_default(dsn: str):
    import psycopg

    return psycopg.connect(dsn)


def enforce_read_only(conn) -> None:
    cur = conn.cursor()
    try:
        cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        cur.execute("SET LOCAL transaction_read_only = on")
        cur.execute("SHOW transaction_read_only")
        rows = cur.fetchall()
    except Exception as exc:
        raise _InvariantFailure("readonly_unenforceable") from exc
    finally:
        cur.close()
    if not rows or str(rows[0][0]).strip().lower() != "on":
        raise _InvariantFailure("readonly_unenforceable")


def _check_role(conn) -> None:
    sql = (
        "SELECT rolsuper, rolcreaterole, rolcreatedb, rolreplication, rolbypassrls "
        "FROM pg_roles WHERE rolname = current_user"
    )
    cur = conn.cursor()
    try:
        cur.execute(sql)
        rows = cur.fetchall()
    except Exception as exc:
        raise _InvariantFailure("role_privileged") from exc
    finally:
        cur.close()
    if not rows:
        raise _InvariantFailure("role_privileged")
    if any(bool(v) for v in rows[0]):
        raise _InvariantFailure("role_privileged")


def _discover_candidate_schemas(conn) -> list:
    sql = (
        "SELECT table_schema FROM information_schema.tables "
        "WHERE table_name = 'workspace' "
        "AND table_schema NOT LIKE 'pg\\_%' AND table_schema <> 'information_schema' "
        "ORDER BY table_schema"
    )
    cur = conn.cursor()
    try:
        cur.execute(sql)
        rows = cur.fetchall()
    except Exception as exc:
        raise _InvariantFailure("schema_none") from exc
    finally:
        cur.close()
    schemas = [str(r[0]) for r in rows if r and r[0] is not None]
    return sorted(set(schemas))


_WRITE_TABLE_PRIVS = ("INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER")
_WRITE_COLUMN_PRIVS = ("INSERT", "UPDATE", "REFERENCES")


def _discover_schema_tables(conn, schemas: list) -> list:
    """Return a flat list of (schema, table) pairs for all candidate schemas."""
    if not schemas:
        return []
    placeholders = ", ".join("%s" for _ in schemas)
    sql = (
        "SELECT n.nspname, c.relname "
        "FROM pg_catalog.pg_class c "
        "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname IN (" + placeholders + ") "
        "AND c.relkind = 'r' "
        "ORDER BY n.nspname, c.relname"
    )
    cur = conn.cursor()
    try:
        cur.execute(sql, tuple(schemas))
        rows = cur.fetchall()
    except Exception:
        return []
    finally:
        cur.close()
    return [(str(r[0]), str(r[1])) for r in rows if r and r[0] and r[1]]


def _check_write_grants(conn, candidate_schemas: list, current_user: str) -> None:
    # Use PostgreSQL's effective-privilege functions so inherited, PUBLIC,
    # and column-level write grants are all caught.  We never execute a
    # write to probe for access; we only ask the privilege-check functions.
    # Function signatures:
    #   has_table_privilege(user, table, privilege)
    #   has_any_column_privilege(user, table, privilege)
    #   has_schema_privilege(user, schema, privilege)
    table_pairs = _discover_schema_tables(conn, candidate_schemas)
    if not table_pairs:
        # No tables found; still check schema CREATE privilege.
        table_refs = []
    else:
        table_refs = [
            _quote_ident(schema) + '.' + _quote_ident(table)
            for schema, table in table_pairs
        ]
    parts = []
    params = []
    for table_ref in table_refs:
        for priv in _WRITE_TABLE_PRIVS:
            parts.append("has_table_privilege(%s, %s, %s)")
            params.append(current_user)
            params.append(table_ref)
            params.append(priv)
    for table_ref in table_refs:
        for priv in _WRITE_COLUMN_PRIVS:
            parts.append("has_any_column_privilege(%s, %s, %s)")
            params.append(current_user)
            params.append(table_ref)
            params.append(priv)
    for schema in candidate_schemas:
        parts.append("has_schema_privilege(%s, %s, %s)")
        params.append(current_user)
        params.append(schema)
        params.append("CREATE")
    if not parts:
        raise _InvariantFailure("write_grant_found")
    select = "SELECT " + ", ".join(parts)
    cur = conn.cursor()
    try:
        cur.execute(select, tuple(params))
        rows = cur.fetchall()
    except Exception as exc:
        raise _InvariantFailure("write_grant_found") from exc
    finally:
        cur.close()
    if not rows:
        raise _InvariantFailure("write_grant_found")
    for value in rows[0]:
        if value:
            raise _InvariantFailure("write_grant_found")


def _current_user(conn) -> str:
    cur = conn.cursor()
    try:
        cur.execute("SELECT current_user")
        rows = cur.fetchall()
    except Exception:
        return ""
    finally:
        cur.close()
    if rows and rows[0] and rows[0][0] is not None:
        return str(rows[0][0])
    return ""


def _resolve_schema(conn, candidate_schemas: list, team: str):
    matches = []
    for schema in candidate_schemas:
        sql = "SELECT team FROM " + _quote_ident(schema) + '."workspace" WHERE team = %s'
        cur = conn.cursor()
        try:
            cur.execute(sql, (team,))
            rows = cur.fetchall()
        except Exception:
            rows = []
        finally:
            cur.close()
        if rows and rows[0] and rows[0][0] is not None and str(rows[0][0]) == team:
            matches.append(schema)
    if len(matches) == 1:
        return matches[0], "ok"
    if not matches:
        return None, "schema_none"
    return None, "schema_ambiguous"


def _channel_ids(conn, schema: str, names: list):
    ids = []
    for name in names:
        sql = (
            "SELECT DISTINCT id FROM "
            + _quote_ident(schema)
            + '."channel" WHERE name = %s'
        )
        cur = conn.cursor()
        try:
            cur.execute(sql, (name,))
            rows = cur.fetchall()
        except Exception:
            rows = []
        finally:
            cur.close()
        if len(rows) != 1 or rows[0] is None or rows[0][0] is None:
            return None
        ids.append(rows[0][0])
    # Verify channel flags from the latest record per channel id.
    if not ids:
        return ids
    flag_sql = (
        "SELECT id, "
        "(convert_from(data, 'UTF8')::jsonb ->> 'is_private')::boolean AS is_private, "
        "(convert_from(data, 'UTF8')::jsonb ->> 'is_im')::boolean AS is_im, "
        "(convert_from(data, 'UTF8')::jsonb ->> 'is_mpim')::boolean AS is_mpim "
        "FROM ("
        "  SELECT DISTINCT ON (id) id, data "
        "  FROM " + _quote_ident(schema) + '."channel" '
        "  WHERE id = ANY(%s) "
        "  ORDER BY id, chunk_id DESC, idx DESC"
        ") AS latest"
    )
    cur = conn.cursor()
    try:
        cur.execute(flag_sql, (list(ids),))
        rows = cur.fetchall()
    except Exception:
        return None
    finally:
        cur.close()
    found = set()
    for row in rows:
        if not row:
            return None
        cid, is_private, is_im, is_mpim = row[0], row[1], row[2], row[3]
        if is_private is not False or is_im is not False or is_mpim is not False:
            return None
        found.add(cid)
    if found != set(ids):
        return None
    return ids


def _window_rows(conn, schema: str, channel_ids: list, now: float, hours: int) -> list:
    upper = float(now)
    lower = upper - hours * 3600.0
    sql = (
        "SELECT DISTINCT ON (channel_id, ts) ts, txt FROM "
        + _quote_ident(schema)
        + '."message" WHERE channel_id = ANY(%s) '
        "AND ts::numeric >= %s AND ts::numeric <= %s "
        "ORDER BY channel_id, ts, chunk_id DESC, idx DESC"
    )
    cur = conn.cursor()
    try:
        cur.execute(sql, (list(channel_ids), lower, upper))
        rows = cur.fetchall()
    except Exception as exc:
        raise _InvariantFailure("read_failed") from exc
    finally:
        cur.close()
    out = []
    for row in rows:
        if not row or len(row) < 2:
            continue
        ts_raw, txt = row[0], row[1]
        try:
            ts = Decimal(str(ts_raw))
        except Exception:
            continue
        if not ts.is_finite():
            continue
        if txt is None:
            txt = ""
        if not isinstance(txt, str):
            continue
        out.append((float(ts), txt))
    return out


def _bounded_report(lines: list, reason: str, result: str) -> list:
    lines = [ln for ln in lines if isinstance(ln, str)]
    lines.append("result=" + result)
    lines.append("reason=" + reason)
    return lines


def _source_coverage(conn, candidate_schemas: list, source: dict):
    schema, code = _resolve_schema(conn, candidate_schemas, source["team"])
    if code != "ok":
        return None, code
    channel_ids = _channel_ids(conn, schema, source["channels"])
    if channel_ids is None:
        return None, "channel_missing"
    return (schema, channel_ids), "ok"


def run_probe(sources, connect, now=None) -> tuple:
    lines = ["opt_in_sources=%d" % len(sources)]
    if not sources:
        return _bounded_report(lines, "no_opt_in", "ok"), EXIT_OK
    if now is None:
        now = time.time()
    dsn = os.environ.get(DSN_ENV, "")
    if not dsn:
        return _bounded_report(lines, "dsn_missing", "inaccessible"), EXIT_INACCESSIBLE
    try:
        conn = connect(dsn)
    except Exception:
        return _bounded_report(lines, "connect_failed", "inaccessible"), EXIT_INACCESSIBLE
    try:
        enforce_read_only(conn)
        _check_role(conn)
        candidate_schemas = _discover_candidate_schemas(conn)
        current_user = _current_user(conn)
        _check_write_grants(conn, candidate_schemas, current_user)
    except _InvariantFailure as failure:
        _close(conn)
        return _bounded_report(lines, failure.code, "inaccessible"), EXIT_INACCESSIBLE
    except Exception:
        _close(conn)
        return _bounded_report(lines, "readonly_unenforceable", "inaccessible"), EXIT_INACCESSIBLE
    reason = "ok"
    all_ok = True
    try:
        for index, source in enumerate(sources, start=1):
            total = len(source["channels"])
            try:
                coverage, code = _source_coverage(conn, candidate_schemas, source)
            except Exception:
                coverage, code = None, "read_failed"
            if code != "ok":
                lines.append(
                    "source=%d status=%s channels=0/%d messages_24h=0 messages_7d=0"
                    % (index, code, total)
                )
                all_ok = False
                if reason == "ok":
                    reason = code
                continue
            schema, channel_ids = coverage
            try:
                rows_24h = _window_rows(conn, schema, channel_ids, now, 24)
                rows_7d = _window_rows(conn, schema, channel_ids, now, 168)
            except _InvariantFailure:
                lines.append(
                    "source=%d status=read_failed channels=%d/%d messages_24h=0 messages_7d=0"
                    % (index, total, total)
                )
                all_ok = False
                if reason == "ok":
                    reason = "read_failed"
                continue
            lines.append(
                "source=%d status=ok channels=%d/%d messages_24h=%d messages_7d=%d"
                % (index, total, total, len(rows_24h), len(rows_7d))
            )
    except Exception:
        all_ok = False
        if reason == "ok":
            reason = "read_failed"
    finally:
        _close(conn)
    result = "ok" if all_ok else "inaccessible"
    if all_ok:
        reason = "ok"
    return _bounded_report(lines, reason, result), EXIT_OK if all_ok else EXIT_INACCESSIBLE


def _bounded_utf8(text: str, limit: int = MAX_SNAPSHOT_BYTES) -> bytes:
    data = text.encode("utf-8")
    if len(data) <= limit:
        return data
    raise ValueError("snapshot exceeds %d bytes" % limit)


def run_snapshot(sources, connect, output_path, now=None) -> tuple:
    lines = ["opt_in_sources=%d" % len(sources)]
    if not sources:
        return _bounded_report(lines, "no_opt_in", "inaccessible"), EXIT_INACCESSIBLE
    if now is None:
        now = time.time()
    dsn = os.environ.get(DSN_ENV, "")
    if not dsn:
        return _bounded_report(lines, "dsn_missing", "inaccessible"), EXIT_INACCESSIBLE
    try:
        conn = connect(dsn)
    except Exception:
        return _bounded_report(lines, "connect_failed", "inaccessible"), EXIT_INACCESSIBLE
    resolved = []
    try:
        enforce_read_only(conn)
        _check_role(conn)
        candidate_schemas = _discover_candidate_schemas(conn)
        current_user = _current_user(conn)
        _check_write_grants(conn, candidate_schemas, current_user)
        for source in sources:
            try:
                coverage, code = _source_coverage(conn, candidate_schemas, source)
            except Exception:
                coverage, code = None, "read_failed"
            if code != "ok":
                _close(conn)
                return _bounded_report(lines, code, "inaccessible"), EXIT_INACCESSIBLE
            resolved.append(coverage)
    except _InvariantFailure as failure:
        _close(conn)
        return _bounded_report(lines, failure.code, "inaccessible"), EXIT_INACCESSIBLE
    except Exception:
        _close(conn)
        return _bounded_report(lines, "readonly_unenforceable", "inaccessible"), EXIT_INACCESSIBLE
    fd = None
    try:
        fd = os.open(str(output_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.fchmod(fd, 0o600)
    except FileExistsError:
        _close(conn)
        return _bounded_report(lines, "output_exists", "inaccessible"), EXIT_INACCESSIBLE
    except OSError:
        _close(conn)
        return _bounded_report(lines, "output_refused", "inaccessible"), EXIT_INACCESSIBLE
    succeeded = False
    conn_closed = False
    payload_len = 0
    total = 0
    reason = "read_failed"
    try:
        collected = []
        for schema, channel_ids in resolved:
            rows = _window_rows(conn, schema, channel_ids, now, SNAPSHOT_WINDOW_HOURS)
            total += len(rows)
            collected.extend(rows)
        _close(conn)
        conn_closed = True
        ordered = sorted(collected, key=lambda item: (item[0], item[1]))
        text = "\n".join(txt for _ts, txt in ordered)
        payload = _bounded_utf8(text)
        payload_len = len(payload)
        offset = 0
        while offset < payload_len:
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("short write")
            offset += written
        os.close(fd)
        fd = None
        succeeded = True
        reason = "ok"
    except ValueError:
        if not conn_closed:
            _close(conn)
        succeeded = False
        reason = "snapshot_too_large"
    except BaseException:
        if not conn_closed:
            _close(conn)
        succeeded = False
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if not succeeded:
            try:
                os.unlink(str(output_path))
            except OSError:
                pass
    if succeeded:
        return (
            _bounded_report(
                lines + ["snapshot=ok bytes=%d messages=%d" % (payload_len, total)],
                "ok",
                "ok",
            ),
            EXIT_OK,
        )
    return _bounded_report(lines, reason, "inaccessible"), EXIT_INACCESSIBLE


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 1 or args[0] not in ("probe", "snapshot"):
        sys.stderr.write("usage: archive_reader.py probe | snapshot --output PATH\n")
        return EXIT_USAGE
    command = args[0]
    try:
        sources = load_slack_sources(default_watchlist_path())
    except Exception:
        sys.stderr.write("watchlist unreadable\n")
        return EXIT_INACCESSIBLE
    if command == "probe":
        lines, code = run_probe(sources, _connect_default)
        for line in lines:
            print(line)
        return code
    output = None
    i = 1
    while i < len(args):
        if args[i] == "--output" and i + 1 < len(args):
            output = args[i + 1]
            i += 2
        else:
            sys.stderr.write("usage: archive_reader.py snapshot --output PATH\n")
            return EXIT_USAGE
    if not output:
        sys.stderr.write("usage: archive_reader.py snapshot --output PATH\n")
        return EXIT_USAGE
    lines, code = run_snapshot(sources, _connect_default, output)
    for line in lines:
        print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
