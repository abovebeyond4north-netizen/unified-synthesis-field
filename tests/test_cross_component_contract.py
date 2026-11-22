"""A1.5/A1.6: public cross-component evidence contract and negative replay.

The resource auditor checks recorded resource fields; the claim preflight
checks paired outcomes and frozen-claim conditions. Neither authenticates
the source of the records or independently establishes matched compute.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.audit_resource_receipts import ARMS, audit_files, audit_report
from usf06.preflight import evaluation_claim_preflight


FROZEN_TASKS = ("T010", "T011")


def synthetic_report():
    rows = []
    for task, family in zip(FROZEN_TASKS, ("quadratic", "cubic")):
        for mode in ARMS:
            rows.append({
                "task_id": task,
                "family": family,
                "mode": mode,
                "proposals": 10,
                "attempts": 15 if mode == "random" else 25,
                "formal_elapsed_s": 0,
                "holdout_exact": int(
                    task == "T010" and mode in ("cold_adaptive", "adaptive")
                ),
            })
    return {
        "config": {"seed": 9981, "test_tasks": 2, "budget_per_task": 160},
        "test_rows": rows,
    }


def preflight(report, task_order=FROZEN_TASKS):
    return evaluation_claim_preflight(
        report["test_rows"],
        expected_test_tasks=report["config"]["test_tasks"],
        configured_budget_per_task=report["config"]["budget_per_task"],
        expected_task_order=task_order,
        bootstrap_seed=1234,
    )


class CrossComponentContractTests(unittest.TestCase):
    def test_clean_shared_rows_agree_but_cannot_promote(self):
        report = synthetic_report()
        audited = audit_report(report)
        claim = preflight(report)
        self.assertEqual(audited["paired_trained_minus_cold_mean"], 0)
        self.assertEqual(claim["paired_mean_difference"], 0)
        for check in (
            "only_required_arms_present",
            "complete_required_arms",
            "unique_task_ids_per_arm",
            "task_ids_paired_across_arms",
            "task_order_matches_frozen_receipt",
            "valid_correctness_outcomes",
            "configured_search_budget_matches_frozen_contract",
        ):
            self.assertTrue(claim["checks"][check], check)
        self.assertFalse(audited["actual_compute_equivalence_established"])
        self.assertFalse(audited["milestone_claim_eligible"])
        self.assertFalse(claim["eligible"])
        self.assertFalse(claim["milestone_a_accepted"])
        self.assertIn(
            "independently_authenticated_provenance",
            claim["milestone_a_blocking_reasons"],
        )

    def test_unexpected_and_duplicate_rows_block_both_readers(self):
        for kind in ("unexpected", "duplicate"):
            report = synthetic_report()
            if kind == "unexpected":
                report["test_rows"][0]["mode"] = "adaptive_failed"
                key = "only_required_arms_present"
            else:
                report["test_rows"][0] = copy.deepcopy(report["test_rows"][1])
                key = "unique_task_ids_per_arm"
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):
                    audit_report(report)
                claim = preflight(report)
                self.assertFalse(claim["checks"][key])
                self.assertFalse(claim["eligible"])
                self.assertFalse(claim["milestone_a_accepted"])

    def test_malformed_exact_outcomes_block_both_readers(self):
        for invalid in (0.5, True, "1", float("nan"), float("inf"),
                        10**400, -(10**400)):
            with self.subTest(value=repr(invalid)[:40]):
                report = synthetic_report()
                report["test_rows"][0]["holdout_exact"] = invalid
                with self.assertRaises(ValueError):
                    audit_report(report)
                claim = preflight(report)
                self.assertFalse(claim["checks"]["valid_correctness_outcomes"])
                self.assertFalse(claim["eligible"])
                self.assertFalse(claim["milestone_a_accepted"])

    def test_unpaired_task_identity_blocks_both_readers(self):
        report = synthetic_report()
        report["test_rows"][0]["task_id"] = "OTHER"
        with self.assertRaises(ValueError):
            audit_report(report)
        claim = preflight(report)
        self.assertFalse(claim["checks"]["task_ids_paired_across_arms"])
        self.assertFalse(claim["milestone_a_accepted"])

    def test_conflicting_family_and_resource_receipts_require_auditor(self):
        for kind in ("conflicting_family", "missing_attempts",
                     "over_budget_proposals", "impossible_attempts"):
            with self.subTest(kind=kind):
                report = synthetic_report()
                row = report["test_rows"][0]
                if kind == "conflicting_family":
                    row["family"] = "inconsistent"
                elif kind == "missing_attempts":
                    del row["attempts"]
                elif kind == "over_budget_proposals":
                    row["proposals"] = 161
                else:
                    row["proposals"] = 10
                    row["attempts"] = 3
                with self.assertRaises(ValueError):
                    audit_report(report)
                # Preflight does not check family or resource counters; its
                # standalone output cannot substitute for the resource audit.
                claim = preflight(report)
                self.assertTrue(claim["checks"]["valid_correctness_outcomes"])
                self.assertFalse(claim["milestone_a_accepted"])

    def test_raw_duplicate_json_key_is_rejected_before_preflight(self):
        clean = json.dumps(synthetic_report())
        tampered = clean.replace(
            '"holdout_exact": 0',
            '"holdout_exact": 1, "holdout_exact": 0',
            1,
        )
        self.assertNotEqual(clean, tampered)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tampered.json"
            path.write_text(tampered, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                audit_files([path])

    def test_immutable_public_negative_runs_agree_across_components(self):
        root = Path(__file__).resolve().parent.parent
        paths = [root / "results_v062.json"] + [
            root / "replications" / f"seed_{seed}.json"
            for seed in range(9001, 9006)
        ]
        self.assertTrue(all(path.is_file() for path in paths),
                        "Public regression fixtures must be present")
        aggregate = audit_files(paths)["aggregate"]
        self.assertEqual(aggregate["runs"], 6)
        self.assertEqual(aggregate["paired_tasks"], 36)
        self.assertEqual(
            aggregate["arm_totals"]["adaptive"]["exact_task_successes"], 12
        )
        self.assertEqual(
            aggregate["arm_totals"]["cold_adaptive"]["exact_task_successes"], 13
        )
        self.assertEqual(
            aggregate["arm_totals"]["adaptive"]["reported_mutation_attempts"],
            18762,
        )
        self.assertEqual(
            aggregate["arm_totals"]["random"]["reported_mutation_attempts"],
            7157,
        )
        self.assertEqual(
            aggregate["paired_trained_minus_cold_mean"], round(-1 / 36, 6)
        )
        self.assertFalse(aggregate["milestones_a_b_c_demonstrated"])
        for path in paths:
            with self.subTest(path=path.name):
                report = json.loads(path.read_text(encoding="utf-8"))
                audited = audit_report(report)
                task_order = [
                    row["task_id"] for row in report["test_rows"]
                    if row["mode"] == "adaptive"
                ]
                claim = preflight(report, task_order)
                self.assertAlmostEqual(
                    claim["paired_mean_difference"],
                    audited["paired_trained_minus_cold_mean"],
                    places=5,
                )
                self.assertFalse(claim["milestone_a_accepted"])
                self.assertFalse(audited["milestone_claim_eligible"])


if __name__ == "__main__":
    unittest.main()
