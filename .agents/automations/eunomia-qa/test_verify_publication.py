"""Regression tests for publication scoping in a shared checkout."""

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("verify_publication.py")
SPEC = importlib.util.spec_from_file_location("verify_publication", MODULE_PATH)
vp = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(vp)


def run(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()


class SharedCheckoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.repo = root / "repo"
        self.remote = root / "remote.git"
        self.repo.mkdir()
        run(root, "init", "--bare", str(self.remote))
        run(self.repo, "init", "-b", "main")
        run(self.repo, "config", "user.name", "QA Test")
        run(self.repo, "config", "user.email", "qa@example.invalid")
        run(self.repo, "remote", "add", "origin", str(self.remote))
        self.docs = self.repo / "docs" / "ebpf-qa"
        self.docs.mkdir(parents=True)
        (self.docs / "index.md").write_text("# Index\n", encoding="utf-8")
        (self.docs / "index.zh.md").write_text("# 索引\n", encoding="utf-8")
        (self.repo / "unrelated.txt").write_text("before\n", encoding="utf-8")
        run(self.repo, "add", ".")
        run(self.repo, "commit", "-m", "baseline")
        run(self.repo, "push", "-u", "origin", "main")
        self.old_repo, self.old_docs = vp.REPO, vp.DOCS
        vp.REPO, vp.DOCS = self.repo, self.docs

    def tearDown(self):
        vp.REPO, vp.DOCS = self.old_repo, self.old_docs
        self.tmp.cleanup()

    def candidate(self):
        stem = "2026-09-06-shared-checkout"
        eng = self.docs / f"{stem}.md"
        zhd = self.docs / f"{stem}.zh.md"
        eng.write_text("# Can this publish?\n", encoding="utf-8")
        zhd.write_text("# 这能发布吗？\n", encoding="utf-8")
        with (self.docs / "index.md").open("a", encoding="utf-8") as out:
            out.write(f"/ebpf-qa/{stem}/\n")
        with (self.docs / "index.zh.md").open("a", encoding="utf-8") as out:
            out.write(f"/zh/ebpf-qa/{stem}/\n")
        return stem, eng, zhd

    def test_candidate_check_ignores_unrelated_dirty_file(self):
        stem, eng, zhd = self.candidate()
        (self.repo / "unrelated.txt").write_text("dirty\n", encoding="utf-8")
        paths = vp.check_candidate_paths(eng, zhd)
        self.assertEqual(paths, [eng, zhd, self.docs / "index.md", self.docs / "index.zh.md"])
        self.assertIn("unrelated.txt", run(self.repo, "status", "--short"))

    def test_clean_candidate_resolves_published_commit(self):
        stem, eng, zhd = self.candidate()
        run(self.repo, "add", "docs/ebpf-qa")
        run(self.repo, "commit", "-m", "publish candidate", "--", "docs/ebpf-qa")
        paths = [eng, zhd, self.docs / "index.md", self.docs / "index.zh.md"]
        self.assertEqual(vp.pending_candidate_changes(paths), "")
        self.assertEqual(
            vp.find_published_commit(eng), run(self.repo, "rev-parse", "HEAD")
        )

    def test_unpublished_candidate_reports_pending(self):
        stem, eng, zhd = self.candidate()
        paths = [eng, zhd, self.docs / "index.md", self.docs / "index.zh.md"]
        self.assertTrue(vp.pending_candidate_changes(paths))
        with self.assertRaisesRegex(vp.Failure, "not committed"):
            vp.find_published_commit(eng)

    def test_multiple_dated_candidates_resolve_to_complete_pair(self):
        stem, eng, zhd = self.candidate()
        (self.docs / f"{stem}-abandoned.md").write_text("# Older one?\n", encoding="utf-8")
        picked_eng, picked_zhd = vp.find_qa_pair("2026-09-06")
        self.assertEqual((picked_eng, picked_zhd), (eng, zhd))

    def test_multiple_complete_candidates_resolve_to_most_recent(self):
        stem, eng, zhd = self.candidate()
        old_eng = self.docs / f"{stem}-old.md"
        old_zhd = self.docs / f"{stem}-old.zh.md"
        old_eng.write_text("# Older question?\n", encoding="utf-8")
        old_zhd.write_text("# 旧问题？\n", encoding="utf-8")
        old_ts = 1_000_000.0
        for path in (old_eng, old_zhd):
            os.utime(path, ns=(int(old_ts * 1e9), int(old_ts * 1e9)))
        picked_eng, _ = vp.find_qa_pair("2026-09-06")
        self.assertEqual(picked_eng, eng)

    def test_scoped_commit_preserves_unrelated_staged_change(self):
        stem, eng, zhd = self.candidate()
        unrelated = self.repo / "unrelated.txt"
        unrelated.write_text("staged but unrelated\n", encoding="utf-8")
        run(self.repo, "add", "unrelated.txt")
        paths = vp.check_candidate_paths(eng, zhd)
        commit = vp.commit_and_push(paths, "shared-checkout", "2026-09-06")
        changed = set(run(self.repo, "show", "--format=", "--name-only", commit).splitlines())
        self.assertEqual(
            changed,
            {
                f"docs/ebpf-qa/{stem}.md",
                f"docs/ebpf-qa/{stem}.zh.md",
                "docs/ebpf-qa/index.md",
                "docs/ebpf-qa/index.zh.md",
            },
        )
        self.assertEqual(run(self.repo, "diff", "--cached", "--name-only"), "unrelated.txt")

    def test_commit_failure_is_reportable(self):
        paths = [self.docs / "index.md", self.docs / "index.zh.md"]
        with self.assertRaisesRegex(vp.Failure, "scoped git commit failed"):
            vp.commit_and_push(paths, "no-change", "2026-09-06")

    def test_public_route_stays_below_static_export(self):
        out = self.repo / "app" / "out"
        self.assertEqual(
            vp.built_route_path(out, "/ebpf-qa/example/"),
            out / "ebpf-qa" / "example" / "index.html",
        )

    def test_public_check_waits_past_short_deployment_race(self):
        body = b"deployed content"
        response = mock.MagicMock()
        response.__enter__.return_value.status = 200
        response.__enter__.return_value.read.return_value = body
        responses = [vp.urllib.error.HTTPError("url", 404, "", {}, None)] * 5
        responses.append(response)
        with (
            mock.patch.object(vp.urllib.request, "urlopen", side_effect=responses),
            mock.patch.object(vp.time, "sleep") as sleep,
        ):
            vp.check_public("/new-route/", "deployed content")
        self.assertEqual(sleep.call_count, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
