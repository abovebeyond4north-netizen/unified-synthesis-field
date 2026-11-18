"""Regression tests for lossless JSON parsing and exact binary aggregation."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tools.audit_resource_receipts import ARMS, audit_files, main


def report(seed, tasks, wins, losses=0):
    return {
        "config": {"seed": seed, "test_tasks": tasks, "budget_per_task": 10},
        "test_rows": [
            {"task_id": f"T{i}", "family": "synthetic", "mode": arm,
             "proposals": 6, "attempts": 0, "formal_elapsed_s": 0,
             "holdout_exact": float((arm == "adaptive" and i < wins)
                                    or (arm == "cold_adaptive" and i >= tasks - losses))}
            for i in range(tasks) for arm in ARMS
        ],
    }


class ResourceReceiptIntegrityTests(unittest.TestCase):
    def test_aggregate_rounds_once_for_both_signs(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.json", Path(tmp) / "b.json"
            a.write_text(json.dumps(report(1, 1, 0)), encoding="utf-8")
            for wins, losses in ((5, 0), (0, 5)):
                b.write_text(json.dumps(report(2, 6, wins, losses)), encoding="utf-8")
                for paths in ([a, b], [b, a]):
                    with self.subTest(wins=wins, losses=losses, paths=paths):
                        result = audit_files(paths)["aggregate"]
                        self.assertEqual(result["paired_tasks"], 7)
                        self.assertEqual(result["paired_trained_minus_cold_mean"], round((wins-losses)/7, 6))
                        self.assertFalse(result["milestones_a_b_c_demonstrated"])

    def test_duplicate_keys_rejected_at_all_levels(self):
        clean = json.dumps(report(1, 1, 0))
        cases = {
            "config": clean.replace('"seed": 1', '"seed": 9, "seed": 1', 1),
            "outcome": clean.replace('"holdout_exact": 0.0', '"holdout_exact": 1.0, "holdout_exact": 0.0', 1),
            "identical": clean.replace('"seed": 1', '"seed": 1, "seed": 1', 1),
            "escaped": clean.replace('"seed": 1', '"s\\u0065ed": 9, "seed": 1', 1),
            "unexamined_nested": clean[:-1] + ', "meta": {"note": 0, "note": 1}}',
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            for label, payload in cases.items():
                with self.subTest(label=label):
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                        audit_files([path])
                    with redirect_stdout(io.StringIO()) as output:
                        self.assertEqual(main([str(path)]), 1)
                    self.assertEqual(json.loads(output.getvalue())["status"], "REJECTED")

    def test_clean_legacy_float_report_remains_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(report(1, 2, 1)), encoding="utf-8")
            self.assertEqual(audit_files([path])["aggregate"]["paired_trained_minus_cold_mean"], 0.5)
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main([str(path)]), 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "STRUCTURALLY_VALID_REPORTED_DATA_ONLY")


if __name__ == "__main__":
    unittest.main()
