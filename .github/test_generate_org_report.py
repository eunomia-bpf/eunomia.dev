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

    @staticmethod
    def _week(day: str, days):
        return {"week": int(REPORT.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=REPORT.timezone.utc).timestamp()), "days": days}

    @patch.object(REPORT, "run_gh_api")
    def test_get_new_stars_counts_days_inside_the_period(self, run_api):
        run_api.side_effect = [
            [
                {
                    "name": "demo",
                    "full_name": "eunomia-bpf/demo",
                    "html_url": "https://github.com/eunomia-bpf/demo",
                    "stargazers_count": 12,
                    "archived": False,
                }
            ],
            [
                self._week("2026-07-06", [0, 0, 0, 0, 0, 0, 1]),
                self._week("2026-07-13", [0, 1, 2, 0, 0, 0, 0]),
                self._week("2026-07-20", [5, 0, 0, 0, 0, 0, 0]),
            ],
        ]

        total, breakdown, current_total = REPORT.get_new_stars(
            "eunomia-bpf", "2026-07-13T00:00:00Z", "2026-07-19T23:59:59Z"
        )

        self.assertEqual(total, 3)
        self.assertEqual(current_total, 12)
        self.assertEqual(len(breakdown), 1)

    @patch.object(REPORT, "run_gh_api")
    def test_get_new_stars_counts_a_partial_week_at_the_period_edges(self, run_api):
        run_api.side_effect = [
            [
                {
                    "name": "demo",
                    "full_name": "eunomia-bpf/demo",
                    "html_url": "https://github.com/eunomia-bpf/demo",
                    "stargazers_count": 12,
                    "archived": False,
                }
            ],
            [self._week("2026-07-13", [1, 1, 1, 1, 1, 1, 1])],
        ]

        total, _breakdown, _current_total = REPORT.get_new_stars(
            "eunomia-bpf", "2026-07-15T00:00:00Z", "2026-07-16T23:59:59Z"
        )

        self.assertEqual(total, 2)

    @patch.object(REPORT, "run_gh_api")
    def test_get_new_stars_rejects_incomplete_results(self, run_api):
        run_api.side_effect = [
            [
                {
                    "name": "demo",
                    "full_name": "eunomia-bpf/demo",
                    "html_url": "https://github.com/eunomia-bpf/demo",
                    "stargazers_count": 12,
                    "archived": False,
                }
            ],
            RuntimeError("rate limited"),
        ]

        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            REPORT.get_new_stars(
                "eunomia-bpf", "2026-07-13T00:00:00Z", "2026-07-19T23:59:59Z"
            )


if __name__ == "__main__":
    unittest.main()
