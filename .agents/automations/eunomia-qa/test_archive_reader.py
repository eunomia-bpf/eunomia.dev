"""Deterministic unittest coverage for archive_reader (fake connections only)."""

import io
import os
import re
import stat
import tempfile
import unittest
from decimal import Decimal
from contextlib import redirect_stdout
from pathlib import Path

import archive_reader as ar

REPO_ROOT = Path(__file__).resolve().parents[3]
WATCHLIST = REPO_ROOT / ".github" / "publisher" / "media" / "community-watchlist.yaml"
TEAM_A = "Team A"
TOKEN_RE = re.compile(r"^([a-z0-9_]+)=([0-9]+(?:/[0-9]+)?|[A-Za-z0-9_]+)$")


class FakeCursor:
    def __init__(self, backend):
        self._backend = backend
        self._rows = []
        self.closed = False

    def execute(self, sql, params=None):
        self._rows = self._backend.handle(sql, params)

    def fetchall(self):
        return self._rows

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, backend):
        self._backend = backend

    def cursor(self):
        return FakeCursor(self._backend)

    def close(self):
        self._backend.connection_closed = True


class FakeBackend:
    def __init__(
        self,
        schema_names=("other_ws", "team_a"),
        team_by_schema=None,
        channels_by_schema=None,
        messages_by_schema=None,
        role_flags=(False, False, False, False, False),
        grant_count=0,
        read_only=True,
        ro_show="on",
        current_user="probe_user",
        message_error=False,
        channel_flags=None,
    ):
        self.schema_names = list(schema_names)
        self.team_by_schema = dict(
            team_by_schema if team_by_schema is not None else {"other_ws": "Some Other Team", "team_a": TEAM_A}
        )
        self.channels_by_schema = dict(
            channels_by_schema if channels_by_schema is not None else {"team_a": {"ebpf": "C_EBPF"}}
        )
        self.messages_by_schema = dict(messages_by_schema or {})
        self.role_flags = tuple(role_flags)
        self.grant_count = grant_count
        self.read_only = read_only
        self.ro_show = ro_show
        self.current_user = current_user
        self.message_error = message_error
        # channel_flags: dict of channel_id -> {"is_private": bool, "is_im": bool, "is_mpim": bool}
        # None means all flags are False (normal public channel)
        self.channel_flags = channel_flags
        self.executed = []
        self.message_results = []
        self.connection_closed = False

    @staticmethod
    def _unquote(ident):
        return ident.replace('""', '"')

    def _schema_of(self, sql, table):
        pattern = r'"((?:[^"]|"")*)"\."' + re.escape(table) + r'"'
        match = re.search(pattern, sql)
        return self._unquote(match.group(1)) if match else None

    def handle(self, sql, params):
        self.executed.append((sql, params))
        if sql.startswith("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY") or sql.startswith(
            "SET LOCAL transaction_read_only"
        ):
            if not self.read_only:
                raise RuntimeError("read-only transactions are disabled in this test")
            return []
        if sql.startswith("SHOW transaction_read_only"):
            return [(self.ro_show,)]
        if "pg_roles" in sql:
            return [tuple(self.role_flags)]
        if "information_schema.tables" in sql:
            return [(name,) for name in self.schema_names]
        if "pg_catalog.pg_class" in sql:
            # Return one representative table per candidate schema
            result = []
            for schema in self.schema_names:
                result.append((schema, "workspace"))
                result.append((schema, "channel"))
                result.append((schema, "message"))
            return result
        if "has_table_privilege" in sql or "has_any_column_privilege" in sql or "has_schema_privilege" in sql:
            # Count the number of has_*_privilege calls in the SQL
            n = sql.count("has_table_privilege") + sql.count("has_any_column_privilege") + sql.count("has_schema_privilege")
            # If grant_count > 0, at least one privilege check returns True
            values = [False] * n
            if self.grant_count > 0:
                values[0] = True
            return [tuple(values)]
        if sql.startswith("SELECT current_user"):
            return [(self.current_user,)]
        if '"workspace"' in sql:
            schema = self._schema_of(sql, "workspace")
            wanted = params[0] if params else None
            team = self.team_by_schema.get(schema)
            if team is not None and team == wanted:
                return [(team,)]
            return []
        if '"channel"' in sql:
            schema = self._schema_of(sql, "channel")
            # Flag-check query: has convert_from in SQL
            if "convert_from" in sql:
                ids = params[0] if params else []
                result = []
                flags_dict = self.channel_flags
                if flags_dict is None:
                    flags_dict = {}
                for cid in ids:
                    if cid not in flags_dict:
                        result.append((cid, None, None, None))
                    else:
                        f = flags_dict[cid]
                        result.append((cid, f.get("is_private", False), f.get("is_im", False), f.get("is_mpim", False)))
                return result
            wanted = params[0] if params else None
            channels = self.channels_by_schema.get(schema, {})
            if wanted in channels:
                return [(channels[wanted],)]
            return []
        if '"message"' in sql:
            if self.message_error:
                raise RuntimeError("archive read exploded")
            schema = self._schema_of(sql, "message")
            ids, lower, upper = params
            lower_d = Decimal(str(lower))
            upper_d = Decimal(str(upper))
            rows = self.messages_by_schema.get(schema, [])
            best = {}
            for row in rows:
                if row["channel_id"] not in ids:
                    continue
                ts_value = Decimal(str(row["ts"]))
                if not lower_d <= ts_value <= upper_d:
                    continue
                key = (row["channel_id"], str(row["ts"]))
                rank = (int(row.get("chunk_id", 0)), int(row.get("idx", 0)))
                if key not in best or rank > best[key][0]:
                    best[key] = (rank, row["ts"], row["txt"])
            result = [
                (ts, txt)
                for (_channel_id, _ts), (_rank, ts, txt) in sorted(
                    best.items(), key=lambda item: (Decimal(str(item[0][1])), item[1][2])
                )
            ]
            self.message_results.append(result)
            return result
        raise AssertionError("unrouted sql: %s" % sql)


def make_connector(backend):
    holder = []

    def connect(dsn):
        holder.append(dsn)
        return FakeConnection(backend)

    return connect, holder


def sources(team=TEAM_A, channels=("ebpf",)):
    return [{"team": team, "channels": list(channels)}]


def expect_bounded(lines):
    for line in lines:
        for token in line.split(" "):
            match = TOKEN_RE.match(token)
            assert match is not None, "unbounded stdout token: %r" % token
            value = match.group(2)
            if not re.match(r"^\d+(?:/\d+)?$", value):
                assert value in ar.REASON_CODES or value in ("ok", "inaccessible"), value


def run_probe_captured(backend, src=None, now=None, with_dsn=True):
    connector, holder = make_connector(backend)
    buf = io.StringIO()
    old = os.environ.get(ar.DSN_ENV)
    if with_dsn:
        os.environ[ar.DSN_ENV] = "DSN_SENTINEL_DO_NOT_PRINT"
    else:
        os.environ.pop(ar.DSN_ENV, None)
    try:
        with redirect_stdout(buf):
            lines, code = ar.run_probe(src if src is not None else sources(), connector, now=now)
    finally:
        if old is None:
            os.environ.pop(ar.DSN_ENV, None)
        else:
            os.environ[ar.DSN_ENV] = old
    return lines, code, buf.getvalue(), holder


class Base(unittest.TestCase):
    def run_probe(self, backend, **kwargs):
        lines, code, out, holder = run_probe_captured(backend, **kwargs)
        expect_bounded(lines)
        return lines, code, out, holder

    def assert_last_line(self, lines, expected):
        self.assertTrue(lines, "no output lines")
        self.assertEqual(lines[-1], expected)


class SchemaResolutionTests(Base):
    def test_exact_schema_probe_ok(self):
        backend = FakeBackend(
            channel_flags={"C_EBPF": {"is_private": False, "is_im": False, "is_mpim": False}},
        )
        lines, code, out, _ = self.run_probe(backend)
        self.assertEqual(code, 0)
        self.assertEqual(lines[0], "opt_in_sources=1")
        self.assertIn("source=1 status=ok channels=1/1 messages_24h=0 messages_7d=0", lines)
        self.assertEqual([ln for ln in lines if ln.startswith("result=")], ["result=ok"])
        self.assert_last_line(lines, "reason=ok")

    def test_no_schema(self):
        backend = FakeBackend(team_by_schema={"other_ws": "Some Other Team", "team_a": "Team B"})
        lines, code, out, _ = self.run_probe(backend)
        self.assertEqual(code, 1)
        self.assertIn("source=1 status=schema_none channels=0/1 messages_24h=0 messages_7d=0", lines)
        self.assertEqual([ln for ln in lines if ln.startswith("result=")], ["result=inaccessible"])
        self.assert_last_line(lines, "reason=schema_none")

    def test_ambiguous_schema(self):
        backend = FakeBackend(
            schema_names=("team_a_copy", "team_a"),
            team_by_schema={"team_a": TEAM_A, "team_a_copy": TEAM_A},
            channels_by_schema={"team_a": {"ebpf": "C_EBPF"}, "team_a_copy": {"ebpf": "C_EBPF_2"}},
        )
        lines, code, out, _ = self.run_probe(backend)
        self.assertEqual(code, 1)
        self.assertIn("source=1 status=schema_ambiguous channels=0/1 messages_24h=0 messages_7d=0", lines)
        self.assert_last_line(lines, "reason=schema_ambiguous")

    def test_missing_channel(self):
        backend = FakeBackend(channels_by_schema={"team_a": {"not-ebpf": "C_OTHER"}})
        lines, code, out, _ = self.run_probe(backend)
        self.assertEqual(code, 1)
        self.assertIn("source=1 status=channel_missing channels=0/1 messages_24h=0 messages_7d=0", lines)
        self.assert_last_line(lines, "reason=channel_missing")

    def test_channel_case_mismatch_is_missing(self):
        backend = FakeBackend(channels_by_schema={"team_a": {"Ebpf": "C_EBPF"}})
        lines, code, out, _ = self.run_probe(backend)
        self.assertEqual(code, 1)
        self.assert_last_line(lines, "reason=channel_missing")

    def test_private_channel_rejected(self):
        backend = FakeBackend(channel_flags={"C_EBPF": {"is_private": True, "is_im": False, "is_mpim": False}})
        lines, code, out, _ = self.run_probe(backend)
        self.assertEqual(code, 1)
        self.assert_last_line(lines, "reason=channel_missing")

    def test_im_channel_rejected(self):
        backend = FakeBackend(channel_flags={"C_EBPF": {"is_private": False, "is_im": True, "is_mpim": False}})
        lines, code, out, _ = self.run_probe(backend)
        self.assertEqual(code, 1)
        self.assert_last_line(lines, "reason=channel_missing")

    def test_mpim_channel_rejected(self):
        backend = FakeBackend(channel_flags={"C_EBPF": {"is_private": False, "is_im": False, "is_mpim": True}})
        lines, code, out, _ = self.run_probe(backend)
        self.assertEqual(code, 1)
        self.assert_last_line(lines, "reason=channel_missing")

    def test_missing_channel_flags_rejected(self):
        # C_EBPF not in channel_flags means the flag query returns no row for it
        backend = FakeBackend(channels_by_schema={"team_a": {"ebpf": "C_EBPF"}})
        backend.channel_flags = {}
        lines, code, out, _ = self.run_probe(backend)
        self.assertEqual(code, 1)
        self.assert_last_line(lines, "reason=channel_missing")

    def test_public_channel_ok(self):
        backend = FakeBackend(channel_flags={"C_EBPF": {"is_private": False, "is_im": False, "is_mpim": False}})
        lines, code, out, _ = self.run_probe(backend)
        self.assertEqual(code, 0)
        self.assertEqual([ln for ln in lines if ln.startswith("result=")], ["result=ok"])


class InvariantRejectionTests(Base):
    def test_readonly_unenforceable_set_fails(self):
        backend = FakeBackend(read_only=False)
        lines, code, out, _ = self.run_probe(backend)
        self.assertEqual(code, 1)
        self.assertEqual([ln for ln in lines if ln.startswith("result=")], ["result=inaccessible"])
        self.assert_last_line(lines, "reason=readonly_unenforceable")

    def test_readonly_unenforceable_show_off(self):
        backend = FakeBackend(ro_show="off")
        lines, code, out, _ = self.run_probe(backend)
        self.assertEqual(code, 1)
        self.assert_last_line(lines, "reason=readonly_unenforceable")

    def test_admin_role_rejected(self):
        for index in range(5):
            flags = [False, False, False, False, False]
            flags[index] = True
            backend = FakeBackend(role_flags=tuple(flags))
            lines, code, out, _ = self.run_probe(backend)
            self.assertEqual(code, 1)
            self.assert_last_line(lines, "reason=role_privileged")

    def test_write_grant_rejected(self):
        backend = FakeBackend(grant_count=1)
        lines, code, out, _ = self.run_probe(backend)
        self.assertEqual(code, 1)
        self.assert_last_line(lines, "reason=write_grant_found")

    def test_grant_check_covers_candidate_schemas(self):
        backend = FakeBackend()
        self.run_probe(backend)
        grants = [params for sql, params in backend.executed if "has_table_privilege" in sql]
        self.assertEqual(len(grants), 1)
        # Params include table refs and user; verify candidate schemas and user are present
        params_str = str(grants[0])
        self.assertIn("other_ws", params_str)
        self.assertIn("team_a", params_str)
        self.assertIn("probe_user", params_str)

    def test_inherited_grant_rejected_via_has_table_privilege(self):
        # Simulate an inherited write grant: has_table_privilege returns True
        backend = FakeBackend(grant_count=1)
        lines, code, out, _ = self.run_probe(backend)
        self.assertEqual(code, 1)
        self.assert_last_line(lines, "reason=write_grant_found")
        # Verify the new SQL was used, not the old role_table_grants
        privilege_sqls = [sql for sql, _ in backend.executed if "has_table_privilege" in sql]
        self.assertTrue(privilege_sqls, "has_table_privilege not used in grant check")
        self.assertNotIn("role_table_grants", " ".join(s for s, _ in backend.executed))

    def test_column_level_write_grant_rejected_via_has_any_column_privilege(self):
        # Simulate a column-level write grant: has_any_column_privilege returns True
        backend = FakeBackend(grant_count=1)
        lines, code, out, _ = self.run_probe(backend)
        self.assertEqual(code, 1)
        self.assert_last_line(lines, "reason=write_grant_found")
        # Verify has_any_column_privilege was included in the SQL
        all_sql = " ".join(s for s, _ in backend.executed)
        self.assertIn("has_any_column_privilege", all_sql)

    def test_create_privilege_on_schema_rejected_via_has_schema_privilege(self):
        # Simulate CREATE privilege on a candidate schema
        backend = FakeBackend(grant_count=1)
        lines, code, out, _ = self.run_probe(backend)
        self.assertEqual(code, 1)
        self.assert_last_line(lines, "reason=write_grant_found")
        # Verify has_schema_privilege was included in the SQL
        all_sql = " ".join(s for s, _ in backend.executed)
        self.assertIn("has_schema_privilege", all_sql)

    def test_no_write_grants_probe_ok(self):
        # No write grants: all privilege checks return False
        backend = FakeBackend(
            grant_count=0,
            channel_flags={"C_EBPF": {"is_private": False, "is_im": False, "is_mpim": False}},
        )
        lines, code, out, _ = self.run_probe(backend)
        self.assertEqual(code, 0)
        self.assertEqual([ln for ln in lines if ln.startswith("result=")], ["result=ok"])

    def test_invariants_run_before_any_data_read(self):
        backend = FakeBackend(
            messages_by_schema={"team_a": [{"channel_id": "C_EBPF", "ts": "1.0", "txt": "x"}]},
            channel_flags={"C_EBPF": {"is_private": False, "is_im": False, "is_mpim": False}},
        )
        self.run_probe(backend, now=1000.0)
        executed = [sql for sql, _ in backend.executed]
        first_message = min(i for i, sql in enumerate(executed) if '"message"' in sql)
        for marker in ("SET SESSION CHARACTERISTICS", "pg_roles", "has_table_privilege"):
            first_marker = min(i for i, sql in enumerate(executed) if marker in sql)
            self.assertLess(first_marker, first_message)


class TimestampWindowTests(Base):
    NOW = 2_000_000_000.0

    def build_backend(self):
        return FakeBackend(
            messages_by_schema={
                "team_a": [
                    {
                        "channel_id": "C_EBPF",
                        "ts": str(self.NOW - 3600.0),
                        "txt": "in-window",
                        "load_time": self.NOW - 10_000.0,
                        "chunk_id": 0,
                        "idx": 0,
                    },
                    {
                        "channel_id": "C_EBPF",
                        "ts": str(self.NOW - 86_400.0 - 3600.0),
                        "txt": "old-ts-new-load",
                        "load_time": self.NOW - 60.0,
                        "chunk_id": 0,
                        "idx": 0,
                    },
                ]
            },
            channel_flags={"C_EBPF": {"is_private": False, "is_im": False, "is_mpim": False}},
        )

    def test_window_filters_slack_ts_not_load_time(self):
        backend = self.build_backend()
        lines, code, out, _ = self.run_probe(backend, now=self.NOW)
        self.assertEqual(code, 0)
        self.assertIn("source=1 status=ok channels=1/1 messages_24h=1 messages_7d=2", lines)
        executed = [sql for sql, _ in backend.executed if '"message"' in sql]
        self.assertEqual(len(executed), 2)
        for sql in executed:
            self.assertNotIn("load", sql)
            self.assertIn("ts::numeric >= %s", sql)
            self.assertIn("ts::numeric <= %s", sql)

    def test_cutoff_parameters_are_numeric(self):
        backend = self.build_backend()
        self.run_probe(backend, now=self.NOW)
        params = [p for sql, p in backend.executed if '"message"' in sql]
        self.assertEqual(params[0][1], self.NOW - 24 * 3600.0)
        self.assertEqual(params[0][2], self.NOW)
        self.assertEqual(params[1][1], self.NOW - 7 * 24 * 3600.0)
        self.assertEqual(params[1][2], self.NOW)

    def test_dedup_per_channel_ts_keeps_newest_archived_revision(self):
        rows = [
            {
                "channel_id": "C_EBPF",
                "ts": "100.0000",
                "txt": "old-revision",
                "chunk_id": 1,
                "idx": 0,
            },
            {
                "channel_id": "C_EBPF",
                "ts": "100.0000",
                "txt": "newer-revision",
                "chunk_id": 2,
                "idx": 0,
            },
            {
                "channel_id": "C_EBPF",
                "ts": "100.0000",
                "txt": "newest-revision",
                "chunk_id": 2,
                "idx": 1,
            },
            {"channel_id": "C_EBPF", "ts": "50.5000", "txt": "earlier", "chunk_id": 0, "idx": 0},
            {
                "channel_id": "C_OTHER",
                "ts": "100.0000",
                "txt": "same-ts-other-channel",
                "chunk_id": 7,
                "idx": 3,
            },
        ]
        backend = FakeBackend(
            messages_by_schema={"team_a": rows},
            channels_by_schema={"team_a": {"ebpf": "C_EBPF", "other": "C_OTHER"}},
            channel_flags={
                "C_EBPF": {"is_private": False, "is_im": False, "is_mpim": False},
                "C_OTHER": {"is_private": False, "is_im": False, "is_mpim": False},
            },
        )
        lines, code, out, _ = self.run_probe(
            backend, src=sources(channels=("ebpf", "other")), now=1000.0
        )
        self.assertEqual(code, 0)
        self.assertIn("source=1 status=ok channels=2/2 messages_24h=3 messages_7d=3", lines)
        emitted = backend.message_results[-1]
        texts = {txt for _ts, txt in emitted}
        self.assertEqual(
            texts, {"newest-revision", "earlier", "same-ts-other-channel"}
        )

    def test_window_sql_dedupes_by_channel_and_ts(self):
        backend = self.build_backend()
        self.run_probe(backend, now=self.NOW)
        executed = [sql for sql, _ in backend.executed if '"message"' in sql]
        self.assertEqual(len(executed), 2)
        for sql in executed:
            self.assertIn("SELECT DISTINCT ON (channel_id, ts) ts, txt FROM", sql)
            self.assertIn("ORDER BY channel_id, ts, chunk_id DESC, idx DESC", sql)


class StdoutRedactionTests(Base):
    def test_probe_failure_prints_no_secrets(self):
        backend = FakeBackend(
            schema_names=("secret_schema",),
            team_by_schema={"secret_schema": "Different Team"},
            messages_by_schema={
                "secret_schema": [{"channel_id": "C_X", "ts": "1.0", "txt": "SECRET_MSG_TEXT_7731"}],
            },
        )
        lines, code, out, holder = self.run_probe(backend, now=1000.0)
        self.assertEqual(code, 1)
        for needle in (
            "SECRET_MSG_TEXT_7731",
            "secret_schema",
            "DSN_SENTINEL_DO_NOT_PRINT",
            "SELECT",
            "ERROR",
            "Traceback",
        ):
            self.assertNotIn(needle, out)
        self.assertEqual(holder, ["DSN_SENTINEL_DO_NOT_PRINT"])

    def test_snapshot_success_prints_no_message_text(self):
        backend = FakeBackend(
            messages_by_schema={"team_a": [{"channel_id": "C_EBPF", "ts": "900.0000", "txt": "SECRET_MSG_TEXT_9911"}]},
            channel_flags={"C_EBPF": {"is_private": False, "is_im": False, "is_mpim": False}},
        )
        connector, holder = make_connector(backend)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "snap.txt"
            lines, code = ar.run_snapshot(sources(), connector, str(out_path), now=1000.0)
            expect_bounded(lines)
            self.assertEqual(code, 0)
            self.assertEqual(out_path.read_text(encoding="utf-8"), "SECRET_MSG_TEXT_9911")
        self.assertNotIn("SECRET_MSG_TEXT_9911", " ".join(lines))


class SnapshotBoundTests(Base):
    def test_overflow_fails_not_truncates(self):
        block = "é" * 500
        rows = [
            {"channel_id": "C_EBPF", "ts": str(1_000_000.0 - float(i)), "txt": block} for i in range(1000)
        ]
        backend = FakeBackend(
            messages_by_schema={"team_a": rows},
            channel_flags={"C_EBPF": {"is_private": False, "is_im": False, "is_mpim": False}},
        )
        connector, holder = make_connector(backend)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "big.txt"
            lines, code = ar.run_snapshot(sources(), connector, str(out_path), now=1_000_000.0)
            self.assertEqual(code, 1)
            self.assert_last_line(lines, "reason=snapshot_too_large")
            self.assertFalse(out_path.exists())

    def test_within_limit_succeeds(self):
        block = "é" * 100
        rows = [
            {"channel_id": "C_EBPF", "ts": str(1_000_000.0 - float(i)), "txt": block} for i in range(500)
        ]
        backend = FakeBackend(
            messages_by_schema={"team_a": rows},
            channel_flags={"C_EBPF": {"is_private": False, "is_im": False, "is_mpim": False}},
        )
        connector, holder = make_connector(backend)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "ok.txt"
            lines, code = ar.run_snapshot(sources(), connector, str(out_path), now=1_000_000.0)
            self.assertEqual(code, 0)
            data = out_path.read_bytes()
            self.assertLessEqual(len(data), ar.MAX_SNAPSHOT_BYTES)
            data.decode("utf-8")
            snap = [ln for ln in lines if ln.startswith("snapshot=ok bytes=")]
            self.assertEqual(snap, ["snapshot=ok bytes=%d messages=500" % len(data)])


class SnapshotFileModeTests(Base):
    def test_creates_file_with_mode_0600(self):
        backend = FakeBackend(
            messages_by_schema={"team_a": [{"channel_id": "C_EBPF", "ts": "900.0000", "txt": "hello"}]},
            channel_flags={"C_EBPF": {"is_private": False, "is_im": False, "is_mpim": False}},
        )
        connector, holder = make_connector(backend)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "mode.txt"
            lines, code = ar.run_snapshot(sources(), connector, str(out_path), now=1000.0)
            self.assertEqual(code, 0)
            mode = stat.S_IMODE(os.stat(str(out_path)).st_mode)
            self.assertEqual(mode, 0o600)
            self.assertEqual(out_path.read_text(encoding="utf-8"), "hello")

    def test_never_overwrites_existing_file(self):
        backend = FakeBackend(
            messages_by_schema={"team_a": [{"channel_id": "C_EBPF", "ts": "900.0000", "txt": "new"}]},
            channel_flags={"C_EBPF": {"is_private": False, "is_im": False, "is_mpim": False}},
        )
        connector, holder = make_connector(backend)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "existing.txt"
            out_path.write_bytes(b"original")
            lines, code = ar.run_snapshot(sources(), connector, str(out_path), now=1000.0)
            self.assertEqual(code, 1)
            self.assert_last_line(lines, "reason=output_exists")
            self.assertEqual(out_path.read_bytes(), b"original")

    def test_partial_cleanup_removes_created_file(self):
        backend = FakeBackend(
            message_error=True,
            channel_flags={"C_EBPF": {"is_private": False, "is_im": False, "is_mpim": False}},
        )
        connector, holder = make_connector(backend)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "partial.txt"
            lines, code = ar.run_snapshot(sources(), connector, str(out_path), now=1000.0)
            self.assertEqual(code, 1)
            self.assert_last_line(lines, "reason=read_failed")
            self.assertFalse(out_path.exists())

    def test_refuse_before_create_when_channel_missing(self):
        backend = FakeBackend(channels_by_schema={"team_a": {}})
        connector, holder = make_connector(backend)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "nocity.txt"
            lines, code = ar.run_snapshot(sources(), connector, str(out_path), now=1000.0)
            self.assertEqual(code, 1)
            self.assert_last_line(lines, "reason=channel_missing")
            self.assertFalse(out_path.exists())

    def test_snapshot_requires_all_sources_covered(self):
        two_sources = [
            {"team": TEAM_A, "channels": ["ebpf"]},
            {"team": "Team B", "channels": ["general"]},
        ]
        backend = FakeBackend(
            team_by_schema={"team_a": TEAM_A, "other_ws": "Some Other Team"},
            channels_by_schema={"team_a": {"ebpf": "C_EBPF"}},
            channel_flags={"C_EBPF": {"is_private": False, "is_im": False, "is_mpim": False}},
        )
        connector, holder = make_connector(backend)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "all.txt"
            lines, code = ar.run_snapshot(two_sources, connector, str(out_path), now=1000.0)
            self.assertEqual(code, 1)
            # Second source (Team B) has no matching schema, so schema_none
            self.assertEqual(lines[-1], "reason=schema_none")
            self.assertFalse(out_path.exists())


class WatchlistTests(Base):
    def test_real_watchlist_parses_with_pyyaml(self):
        import yaml

        with open(WATCHLIST, "r", encoding="utf-8") as handle:
            doc = yaml.safe_load(handle)
        slack = [
            s for s in doc["sources"] if s.get("surface") == "Slack" and s.get("archive_opt_in") is True
        ]
        teams = {s["id"]: s["workspace"] for s in slack}
        self.assertEqual(teams["cilium-ebpf-slack"], "Cilium & eBPF")
        self.assertEqual(teams["cncf-slack"], "Cloud Native Computing Foundation")
        loaded = ar.load_slack_sources(WATCHLIST)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(
            {(s["team"], tuple(s["channels"])) for s in loaded},
            {
                ("Cilium & eBPF", ("ebpf",)),
                (
                    "Cloud Native Computing Foundation",
                    ("otel-ebpf-instrumentation", "otel-genai-instrumentation"),
                ),
            },
        )

    def test_dsn_missing_is_inaccessible(self):
        lines, code, out, _ = run_probe_captured(FakeBackend(), with_dsn=False)
        expect_bounded(lines)
        self.assertEqual(code, 1)
        self.assert_last_line(lines, "reason=dsn_missing")

    def test_empty_channels_source_excluded_by_loader(self):
        import yaml
        watchlist_path = Path(tempfile.mkdtemp()) / "watchlist.yaml"
        watchlist_path.write_text(yaml.safe_dump({
            "sources": [
                {"surface": "Slack", "archive_opt_in": True, "workspace": TEAM_A, "channels": []},
                {"surface": "Slack", "archive_opt_in": True, "workspace": TEAM_A, "channels": ["ebpf"]},
            ]
        }), encoding="utf-8")
        loaded = ar.load_slack_sources(watchlist_path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["channels"], ["ebpf"])

    def test_no_opt_in_sources(self):
        lines, code, out, holder = run_probe_captured(FakeBackend(), src=[])
        expect_bounded(lines)
        self.assertEqual(code, 0)
        self.assertEqual(lines[0], "opt_in_sources=0")
        self.assert_last_line(lines, "reason=no_opt_in")
        self.assertEqual(holder, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
