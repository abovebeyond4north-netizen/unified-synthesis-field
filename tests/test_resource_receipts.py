"""Isolated deterministic controls for the read-only USF resource auditor."""
import copy
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

from tools.audit_resource_receipts import audit_files, audit_report, main


def fixture():
    rows = []
    for task, family in (("T010", "quadratic"), ("T011", "cubic")):
        for mode in ("random", "evolution", "cold_adaptive", "adaptive"):
            rows.append({"task_id": task, "family": family, "mode": mode,
                         "proposals": 10, "attempts": 15 if mode == "random" else 25,
                         "formal_elapsed_s": 0,
                         "holdout_exact": int(task == "T010" and mode == "adaptive")})
    return {"config": {"seed": 1, "test_tasks": 2, "budget_per_task": 10}, "test_rows": rows}


class AuditResourceReceiptTests(unittest.TestCase):
    def test_unequal_attempts_are_reported_not_hidden_by_equal_proposals(self):
        receipt = audit_report(fixture())
        self.assertEqual(receipt["arms"]["adaptive"]["reported_proposals"], 20)
        self.assertEqual(receipt["arms"]["random"]["reported_proposals"], 20)
        self.assertEqual(receipt["arms"]["adaptive"]["reported_mutation_attempts"], 50)
        self.assertEqual(receipt["arms"]["random"]["reported_mutation_attempts"], 30)
        self.assertEqual(receipt["paired_trained_minus_cold_mean"], .5)
        self.assertFalse(receipt["milestone_claim_eligible"])
        self.assertFalse(receipt["reported_setup_inclusive_wall_complete"])

    def test_reject_unexpected_modes_and_duplicate_rows(self):
        for alteration in ("unknown", "duplicate"):
            data = fixture()
            if alteration == "unknown":
                data["test_rows"][0]["mode"] = "adaptive_failed"
            else:
                data["test_rows"][0] = copy.deepcopy(data["test_rows"][1])
            with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                audit_report(data)

    def test_exact_outcome_is_strictly_binary(self):
        for accepted in (0, 1, 0.0, 1.0):
            data = fixture()
            data["test_rows"][0]["holdout_exact"] = accepted
            audit_report(data)
        for rejected in (0.5, True, False, "1", -1, 2):
            data = fixture()
            data["test_rows"][0]["holdout_exact"] = rejected
            with self.subTest(value=repr(rejected)), self.assertRaises(ValueError):
                audit_report(data)

    def test_reject_malformed_numeric_and_unpaired_evidence(self):
        for field, value in (("attempts", -1), ("proposals", 11),
                             ("holdout_exact", float("nan")),
                             ("formal_elapsed_s", float("inf")),
                             ("attempts", True), ("proposals", 10 ** 400),
                             ("holdout_exact", 10 ** 400),
                             ("formal_elapsed_s", 10 ** 400)):
            data = fixture()
            data["test_rows"][0][field] = value
            with self.subTest(field=field, value=str(value)[:30]), self.assertRaises(ValueError):
                audit_report(data)
        data = fixture()
        data["test_rows"][0]["task_id"] = "UNPAIRED"
        with self.assertRaises(ValueError):
            audit_report(data)

    def test_attempts_cover_non_initial_proposals(self):
        data = fixture()
        for row in data["test_rows"]:
            row["proposals"] = 6
            row["attempts"] = 0
        audit_report(data)

        data["test_rows"][0]["proposals"] = 10
        data["test_rows"][0]["attempts"] = 3
        with self.assertRaisesRegex(
            ValueError, "attempts cannot produce reported proposals"
        ):
            audit_report(data)

    def test_incomplete_wall_receipt_cannot_be_treated_as_measured_compute(self):
        data = fixture()
        data["test_rows"][0]["setup_inclusive_wall_s"] = .5
        self.assertFalse(audit_report(data)["reported_setup_inclusive_wall_complete"])
        for row in data["test_rows"]:
            row["setup_inclusive_wall_s"] = .5
        receipt = audit_report(data)
        self.assertTrue(receipt["reported_setup_inclusive_wall_complete"])
        self.assertEqual(receipt["arms"]["random"]["reported_setup_inclusive_wall_seconds"], 1)
        self.assertFalse(receipt["actual_compute_equivalence_established"])
        self.assertFalse(receipt["independent_provenance_authenticated"])

    def test_public_six_run_artifacts_preserve_negative_results(self):
        root = Path(__file__).resolve().parent.parent
        paths = [root / "results_v062.json"] + [
            root / "replications" / f"seed_{seed}.json" for seed in range(9001, 9006)
        ]
        if not all(path.exists() for path in paths):
            self.skipTest("public artifacts absent from isolated local test fixture")
        agg = audit_files(paths)["aggregate"]
        self.assertEqual(agg["paired_tasks"], 36)
        self.assertEqual(agg["arm_totals"]["adaptive"], {
            "reported_proposals": 5760,
            "reported_mutation_attempts": 18762,
            "exact_task_successes": 12,
        })
        self.assertEqual(agg["arm_totals"]["cold_adaptive"]["exact_task_successes"], 13)
        self.assertEqual(agg["arm_totals"]["random"]["reported_mutation_attempts"], 7157)
        self.assertEqual(agg["arm_totals"]["evolution"]["reported_mutation_attempts"], 18748)
        self.assertAlmostEqual(agg["paired_trained_minus_cold_mean"], -1 / 36, places=5)
        self.assertFalse(agg["milestones_a_b_c_demonstrated"])

    def test_files_aggregate_and_reject_duplicate_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            one, two = Path(temp) / "one.json", Path(temp) / "two.json"
            one.write_text(json.dumps(fixture()))
            second = fixture()
            second["config"]["seed"] = 2
            two.write_text(json.dumps(second))
            results = audit_files([one, two])
            self.assertEqual(results["aggregate"]["paired_tasks"], 4)
            self.assertEqual(results["aggregate"]["paired_trained_minus_cold_mean"], .5)
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main([str(one), str(two)]), 0)
            self.assertIn("STRUCTURALLY_VALID_REPORTED_DATA_ONLY", output.getvalue())
            with self.assertRaises(ValueError):
                audit_files([one, one])


if __name__ == "__main__":
    unittest.main()

