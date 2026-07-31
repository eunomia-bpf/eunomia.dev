import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("generate_org_report.py")
SPEC = importlib.util.spec_from_file_location("generate_org_report", MODULE_PATH)
REPORT = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(REPORT)


class OrgReportTests(unittest.TestCase):
    def test_parse_paginated_output_merges_pages(self):
        output = "\n".join([json.dumps([{"id": 1}]), json.dumps([{"id": 2}])])
        self.assertEqual(REPORT.parse_gh_api_output(output, paginate=True), [{"id": 1}, {"id": 2}])

    @patch.object(REPORT, "run_gh_api_with_header")
    @patch.object(REPORT, "run_gh_api")
    def test_get_new_stars_returns_period_and_current_totals(self, run_api, run_api_with_header):
        run_api.return_value = [
            {
                "name": "demo",
                "full_name": "eunomia-bpf/demo",
                "html_url": "https://github.com/eunomia-bpf/demo",
                "stargazers_count": 12,
                "archived": False,
            }
        ]
        run_api_with_header.return_value = [
            {"starred_at": "2026-07-14T00:00:00Z"},
            {"starred_at": "2026-07-20T00:00:00Z"},
        ]

        total, breakdown, current_total = REPORT.get_new_stars(
            "eunomia-bpf", "2026-07-13T00:00:00Z", "2026-07-19T23:59:59Z"
        )

        self.assertEqual(total, 1)
        self.assertEqual(current_total, 12)
        self.assertEqual(len(breakdown), 1)

    @patch.object(REPORT, "run_gh_api_with_header", side_effect=RuntimeError("rate limited"))
    @patch.object(REPORT, "run_gh_api")
    def test_get_new_stars_rejects_incomplete_results(self, run_api, _run_api_with_header):
        run_api.return_value = [
            {
                "name": "demo",
                "full_name": "eunomia-bpf/demo",
                "html_url": "https://github.com/eunomia-bpf/demo",
                "stargazers_count": 12,
                "archived": False,
            }
        ]

        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            REPORT.get_new_stars(
                "eunomia-bpf", "2026-07-13T00:00:00Z", "2026-07-19T23:59:59Z"
            )


if __name__ == "__main__":
    unittest.main()
